import os
import pickle
import torch
import time
import pandas as pd
from data.dataset import JSSPDataset
from models.Hetero_actor_critic_ppo import ActorCriticPPO
from env.jssp_environment import JSSPEnvironment
from cp.cp_solver import solve_with_rl_warmstart, solve_instance_your_version
from utils.features import make_hetero_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def get_rl_schedule(instance, model):
    times = torch.tensor(instance['times']).to(device)
    machines = torch.tensor(instance['machines']).to(device)
    env = JSSPEnvironment(times, machines, device=device)
    env.reset()
    
    done = False
    step_count = 0
    while not done:
        data = make_hetero_data(env, device)
        logits, _ = model(data)
        
        avail = env.get_available_actions()
        mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=device)
        mask[avail] = True
        logits = logits.masked_fill(~mask, -1e10)
        
        action = torch.argmax(logits).item()
        _, _, done, _ = env.step(action)
        step_count += 1
        if step_count % 50 == 0:
            print(".", end="", flush=True)
            
    print(" (RL Done)", flush=True)
    return env.get_makespan(), env.extract_job_assignments()

def run_hybrid_experiment(checkpoint_path, data_path, time_limit=30, fix_ratio=0.0):
    print(f"\n--- [Hybrid Experiment] Warm-starting CP with RL ---")
    print(f"Model: {os.path.basename(checkpoint_path)}")
    print(f"Data: {os.path.basename(data_path)}")
    print(f"Fix Ratio: {fix_ratio}")
    
    # 1. Load Model
    model = ActorCriticPPO(
        node_input_dim=3,
        gnn_hidden_dim=64,
        gnn_output_dim=32,
        actor_hidden_dim=32,
        critic_hidden_dim=32
    ).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    
    # 2. Load Data
    with open(data_path, "rb") as f:
        instances = pickle.load(f)
    
    # Cap test size for speed if many instances
    instances = instances[:10] 
    
    results = []
    
    for i, inst in enumerate(instances):
        print(f"\nInstance {i+1}/{len(instances)} (Size: {inst['times'].shape[0]}x{inst['times'].shape[1]})", flush=True)
        
        # A. pure RL
        print("  [RL] Rolling out...", end=" ", flush=True)
        start_rl = time.time()
        rl_makespan, rl_sched = get_rl_schedule(inst, model)
        rl_time = time.time() - start_rl
        print(f"  [RL Result] Makespan: {rl_makespan:.1f} | Time: {rl_time:.2f}s")
        
        # B. pure CP (Cold start)
        print(f"  [CP Cold] Solving ({time_limit}s limit)...", end=" ", flush=True)
        start_cp = time.time()
        cp_cold_ms, _ = solve_instance_your_version(inst['times'], inst['machines'], time_limit_s=time_limit)
        cp_cold_time = time.time() - start_cp
        print(f"Done. Makespan: {cp_cold_ms} | Time: {cp_cold_time:.2f}s")
        
        # C. Hybrid (Warm start)
        print(f"  [Hybrid]  Solving ({time_limit}s limit)...", end=" ", flush=True)
        start_hybrid = time.time()
        cp_warm_ms, _ = solve_with_rl_warmstart(
            inst['times'], inst['machines'], 
            rl_schedule=rl_sched, 
            time_limit_s=time_limit,
            fix_ratio=fix_ratio
        )
        hybrid_time = time.time() - start_hybrid
        print(f"Done. Makespan: {cp_warm_ms} | Time: {hybrid_time:.2f}s")
        
        results.append({
            "id": i,
            "size": f"{inst['times'].shape[0]}x{inst['times'].shape[1]}",
            "rl_ms": rl_makespan,
            "cp_cold_ms": cp_cold_ms,
            "cp_warm_ms": cp_warm_ms,
            "rl_time": rl_time,
            "cp_cold_time": cp_cold_time,
            "hybrid_time": hybrid_time
        })
        
    df = pd.DataFrame(results)
    out_csv = "eval/hybrid_experiment_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[Finished] Results saved to {out_csv}")
    
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    # Assuming we have a Phase_3 (15x15) or later checkpoint
    ckpt = os.path.join(base_dir, "outputs", "models", "best_ppo_latest.pt")
    data = os.path.join(base_dir, "saved", "taillard_mixed_instances.pkl")
    
    if os.path.exists(ckpt) and os.path.exists(data):
        run_hybrid_experiment(ckpt, data, time_limit=20, fix_ratio=0.4)
    else:
        print("Required files (checkpoint or taillard data) not found. Run training first.")
