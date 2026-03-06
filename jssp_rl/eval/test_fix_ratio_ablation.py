import os
import pickle
import torch
import pandas as pd
import time
from models.Hetero_actor_critic_ppo import ActorCriticPPO
from env.jssp_environment import JSSPEnvironment
from cp.cp_solver import solve_with_rl_warmstart
from utils.features import make_hetero_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def get_rl_schedule(instance, model):
    times = instance['times'] if isinstance(instance['times'], torch.Tensor) else torch.tensor(instance['times'])
    machines = instance['machines'] if isinstance(instance['machines'], torch.Tensor) else torch.tensor(instance['machines'])
    times, machines = times.to(device), machines.to(device)
    env = JSSPEnvironment(times, machines, device=device, use_shaping_rewards=False)
    env.reset()
    
    # Pre-calculate static graph topology once per instance
    from models.gin import HeteroGATv2
    static_edge_index = HeteroGATv2.build_edge_index_dict(env.machines)

    done = False
    ops_scheduled = 0
    total_ops = env.num_jobs * env.num_machines
    while not done:
        data = make_hetero_data(env, device, precomputed_edge_index=static_edge_index)
        logits, _ = model(data)
        avail = env.get_available_actions()
        mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=device)
        mask[avail] = True
        logits = logits.masked_fill(~mask, -1e10)
        action = torch.argmax(logits).item()
        _, _, done, _ = env.step(action)
        ops_scheduled += 1
    
    print(f" (RL Done: {ops_scheduled}/{total_ops} ops)", flush=True)
    return env.get_makespan(), env.extract_job_assignments()

def run_ablation(checkpoint_path, fix_ratios=[0.0, 0.1, 0.2, 0.3, 0.4], benchmark_name="Taillard", num_instances=5):
    print(f"Starting Ablation Study on {benchmark_name} (Fix Ratios: {fix_ratios})")
    
    # Load Model
    model = ActorCriticPPO(3, 64, 32, 32, 32).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()

    # Load Instances
    base_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base_dir, "saved", "benchmark_taillard.pkl")
    with open(path, "rb") as f:
        instances = pickle.load(f)[:num_instances]

    results = []

    for inst in instances:
        name = inst['name']
        print(f"Evaluating {name}...")
        rl_ms, rl_sched = get_rl_schedule(inst, model)
        times_np = inst['times'].numpy() if isinstance(inst['times'], torch.Tensor) else inst['times']
        machines_np = inst['machines'].numpy() if isinstance(inst['machines'], torch.Tensor) else inst['machines']

        for ratio in fix_ratios:
            start = time.time()
            # Hardcoded 30s limit for ablation comparison
            ms, _ = solve_with_rl_warmstart(times_np, machines_np, rl_sched, time_limit_s=30.0, fix_ratio=ratio)
            dur = time.time() - start
            results.append({
                "instance": name,
                "fix_ratio": ratio,
                "makespan": ms,
                "time": round(dur, 2),
                "rl_makespan": rl_ms
            })

    df = pd.DataFrame(results)
    out_path = os.path.join(base_dir, "eval", "ablation_fix_ratio_results.csv")
    df.to_csv(out_path, index=False)
    
    print("\nSummary Table (Average Makespan and Time):")
    summary = df.groupby("fix_ratio").agg({"makespan": "mean", "time": "mean"}).reset_index()
    print(summary.to_string())
    return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    ckpt = os.path.join(base_dir, "outputs", "models", "best_ppo_Phase_5.pt")
    if os.path.exists(ckpt):
        run_ablation(ckpt)
    else:
        print("Checkpoint not found.")
