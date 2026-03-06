import os
import pickle
import torch
import time
import pandas as pd
from models.Hetero_actor_critic_ppo import ActorCriticPPO
from env.jssp_environment import JSSPEnvironment
from cp.cp_solver import solve_with_rl_warmstart, solve_instance_your_version
from utils.features import make_hetero_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def get_rl_schedule(instance, model):
    times = instance['times'] if isinstance(instance['times'], torch.Tensor) else torch.tensor(instance['times'])
    machines = instance['machines'] if isinstance(instance['machines'], torch.Tensor) else torch.tensor(instance['machines'])
    times = times.to(device)
    machines = machines.to(device)
    env = JSSPEnvironment(times, machines, device=device, use_shaping_rewards=False)
    env.reset()
    
    from models.gin import HeteroGATv2
    static_edge_index = HeteroGATv2.build_edge_index_dict(env.machines)

    done = False
    while not done:
        data = make_hetero_data(env, device, precomputed_edge_index=static_edge_index)
        logits, _ = model(data)
        avail = env.get_available_actions()
        mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=device)
        mask[avail] = True
        logits = logits.masked_fill(~mask, -1e10)
        action = torch.argmax(logits).item()
        _, _, done, _ = env.step(action)
    return env.get_makespan(), env.extract_job_assignments()

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, "outputs", "models", "best_ppo_latest.pt")
    
    if not os.path.exists(model_path):
        print("Model not found!")
        return

    model = ActorCriticPPO(3, 64, 32, 32, 32).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()

    # Define instances to test (one per size)
    # FT: benchmark_ft.pkl (ft06, ft10)
    # Taillard: benchmark_taillard.pkl (ta01=15x15, ta11=20x15, ta21=20x20, ta31=30x15, ta41=30x20)
    
    test_cases = [
        ("FT", "benchmark_ft.pkl", 0),  # ft06 (6x6)
        ("FT", "benchmark_ft.pkl", 1),  # ft10 (10x10)
        ("Taillard", "benchmark_taillard.pkl", 0),  # ta01 (15x15)
        ("Taillard", "benchmark_taillard.pkl", 10), # ta11 (20x15)
        ("Taillard", "benchmark_taillard.pkl", 20), # ta21 (20x20)
        ("Taillard", "benchmark_taillard.pkl", 30), # ta31 (30x15)
        ("Taillard", "benchmark_taillard.pkl", 40), # ta41 (30x20)
    ]

    results = []
    
    for bm_name, filename, idx in test_cases:
        path = os.path.join(base_dir, "saved", filename)
        with open(path, "rb") as f:
            instances = pickle.load(f)
        inst = instances[idx]
        name = inst.get("name", f"{bm_name}_{idx}")
        times = inst['times']
        machines = inst['machines']
        N, M = times.shape
        bks = inst.get("bks", -1)
        
        tl = 30 # standard 30s limit
        print(f"\nEvaluating {name} ({N}x{M})...")
        
        # RL
        start = time.time()
        rl_ms, rl_sched = get_rl_schedule(inst, model)
        rl_time = time.time() - start
        
        # CP
        start = time.time()
        cp_ms, _ = solve_instance_your_version(times.numpy() if hasattr(times, "numpy") else times, 
                                                machines.numpy() if hasattr(machines, "numpy") else machines, 
                                                time_limit_s=tl)
        cp_time = time.time() - start
        
        # Hybrid
        start = time.time()
        hybrid_ms, _ = solve_with_rl_warmstart(times.numpy() if hasattr(times, "numpy") else times, 
                                                machines.numpy() if hasattr(machines, "numpy") else machines,
                                                rl_schedule=rl_sched, 
                                                time_limit_s=tl, 
                                                fix_ratio=0.4)
        hybrid_time = time.time() - start
        
        results.append({
            "Instance": name,
            "Size": f"{N}x{M}",
            "BKS": bks,
            "RL": rl_ms,
            "CP": cp_ms,
            "Hybrid": hybrid_ms,
            "RL_Time": round(rl_time, 2),
            "CP_Time": round(cp_time, 2),
            "Hybrid_Time": round(hybrid_time, 2),
            "Gap_RL": round((rl_ms - bks)/bks*100, 1) if bks > 0 else "N/A",
            "Gap_Hybrid": round((hybrid_ms - bks)/bks*100, 1) if bks > 0 else "N/A"
        })
        print(f"  RL: {rl_ms} | CP: {cp_ms} | Hybrid: {hybrid_ms}")

    df = pd.DataFrame(results)
    out_path = os.path.join(base_dir, "eval", "quick_compare_results.csv")
    df.to_csv(out_path, index=False)
    print("\nFinal Results Table:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
