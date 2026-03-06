import os
import pickle
import torch
import pandas as pd
import time
from models.Hetero_actor_critic_ppo import ActorCriticPPO
from env.jssp_environment import JSSPEnvironment
from cp.cp_solver import solve_with_rl_warmstart, solve_instance_your_version
from utils.features import prepare_features
from torch_geometric.data import HeteroData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def rollout_instance_with_k(instance, model, k_samples=1):
    times = instance['times'] if isinstance(instance['times'], torch.Tensor) else torch.tensor(instance['times'])
    machines = instance['machines'] if isinstance(instance['machines'], torch.Tensor) else torch.tensor(instance['machines'])
    times, machines = times.to(device), machines.to(device)
    
    from models.gin import HeteroGATv2
    edge_index_dict = HeteroGATv2.build_edge_index_dict(machines)
    job_edges = edge_index_dict[('op', 'job', 'op')].to(device)
    mach_edges = edge_index_dict[('op', 'machine', 'op')].to(device)

    best_ms = float('inf')
    best_sched = None

    for i in range(k_samples):
        # Optimization: use_shaping_rewards=False makes rollout very fast
        env = JSSPEnvironment(times, machines, device=device, use_shaping_rewards=False)
        env.reset()
        done = False
        is_stochastic = (i > 0)
        
        while not done:
            x = prepare_features(env, device)
            data = HeteroData()
            data['op'].x = x
            data['op', 'job', 'op'].edge_index = job_edges
            data['op', 'machine', 'op'].edge_index = mach_edges

            logits, _ = model(data)
            avail = env.get_available_actions()
            mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=device)
            mask[avail] = True
            logits = logits.masked_fill(~mask, -1e10)

            if is_stochastic:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
            else:
                action = torch.argmax(logits).item()
            
            _, _, done, _ = env.step(action)
        
        ms = env.get_makespan()
        if ms < best_ms:
            best_ms = ms
            best_sched = env.extract_job_assignments()

    return best_ms, best_sched

def run_comprehensive_ablation(checkpoint_path, k_values=[1, 5], fix_ratios=[0.0, 0.1, 0.2, 0.4]):
    print(f"Starting Comprehensive Ablation (20x20)")
    print(f"K-values: {k_values} | Fix Ratios: {fix_ratios}")
    
    model = ActorCriticPPO(3, 64, 32, 32, 32).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()

    base_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base_dir, "saved", "benchmark_taillard.pkl")
    with open(path, "rb") as f:
        instances = pickle.load(f)

    test_indices = [20, 21, 22] # ta21-ta23
    results = []

    for idx in test_indices:
        inst = instances[idx]
        name = inst['name']
        times_np = inst['times'].numpy() if isinstance(inst['times'], torch.Tensor) else inst['times']
        machines_np = inst['machines'].numpy() if isinstance(inst['machines'], torch.Tensor) else inst['machines']
        
        print(f"\n>>> Evaluating {name} (20x20) <<<")
        
        # 1. Pure CP Baseline
        print(f"  [1/3] Running Pure CP Baseline (30s)...", end=" ", flush=True)
        start_cp = time.time()
        cp_ms, _ = solve_instance_your_version(times_np, machines_np, time_limit_s=30.0)
        dur_cp = time.time() - start_cp
        print(f"MS: {cp_ms} ({dur_cp:.2f}s)")
        results.append({
            "instance": name, "method": "Pure CP", "k": "-", "fix_ratio": "-", 
            "makespan": cp_ms, "solve_time": round(dur_cp, 2)
        })

        # 2. RL Best-of-K and Hybrids
        for k in k_values:
            print(f"  [2/3] RL Best-of-{k}...", end=" ", flush=True)
            rl_ms, rl_sched = rollout_instance_with_k(inst, model, k_samples=k)
            print(f"MS: {rl_ms}")
            results.append({
                "instance": name, "method": f"RL (K={k})", "k": k, "fix_ratio": "-", 
                "makespan": rl_ms, "solve_time": 0.0 # RL time is small now
            })

            print(f"  [3/3] Hybrid CP (K={k})...")
            for ratio in fix_ratios:
                print(f"    Ratio {ratio*100:.0f}%...", end=" ", flush=True)
                start_h = time.time()
                h_ms, _ = solve_with_rl_warmstart(times_np, machines_np, rl_sched, time_limit_s=30.0, fix_ratio=ratio)
                dur_h = time.time() - start_h
                print(f"MS: {h_ms} ({dur_h:.2f}s)")
                results.append({
                    "instance": name, "method": f"Hybrid (K={k})", "k": k, "fix_ratio": ratio, 
                    "makespan": h_ms, "solve_time": round(dur_h, 2)
                })

    df = pd.DataFrame(results)
    out_path = os.path.join(base_dir, "eval", "comprehensive_20x20_results.csv")
    df.to_csv(out_path, index=False)
    
    print("\nSummary Results (Averaged over instances):")
    summary = df.groupby(["method", "fix_ratio", "k"]).agg({"makespan": "mean", "solve_time": "mean"}).reset_index()
    print(summary.to_string())
    return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    ckpt = os.path.join(base_dir, "outputs", "models", "best_ppo_Phase_5.pt")
    if os.path.exists(ckpt):
        run_comprehensive_ablation(ckpt)
    else:
        print("Checkpoint not found.")
