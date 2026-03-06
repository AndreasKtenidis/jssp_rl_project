import os
import pickle
import torch
import pandas as pd
import time
from models.Hetero_actor_critic_ppo import ActorCriticPPO
from env.jssp_environment import JSSPEnvironment
from cp.cp_solver import solve_with_rl_warmstart
from utils.features import prepare_features
from torch_geometric.data import HeteroData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def rollout_instance_with_k(instance, model, k_samples=1):
    times = instance['times'] if isinstance(instance['times'], torch.Tensor) else torch.tensor(instance['times'])
    machines = instance['machines'] if isinstance(instance['machines'], torch.Tensor) else torch.tensor(instance['machines'])
    times, machines = times.to(device), machines.to(device)
    
    # Pre-calculate static edges for all samples
    from models.gin import HeteroGATv2
    edge_index_dict = HeteroGATv2.build_edge_index_dict(machines)
    job_edges = edge_index_dict[('op', 'job', 'op')].to(device)
    mach_edges = edge_index_dict[('op', 'machine', 'op')].to(device)

    best_ms = float('inf')
    best_sched = None

    for i in range(k_samples):
        env = JSSPEnvironment(times, machines, device=device, use_shaping_rewards=False)
        env.reset()
        done = False
        stochastic = (i > 0) or (k_samples > 1 and i == 0 and False) # Force greedy for first sample if k > 1? 
        # Actually let's just do greedy for i=0 and stochastic for i > 0 if k > 1.
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
        
        if k_samples > 1 and (i+1) % 5 == 0:
            print(".", end="", flush=True)

    return best_ms, best_sched

def run_best_of_k_ablation(checkpoint_path, k_values=[1, 10, 20], fix_ratios=[0.0, 0.1, 0.2, 0.4]):
    print(f"Starting Best-of-K Ablation (20x20) | K={k_values} | Ratios={fix_ratios}")
    
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
        print(f"\nEvaluating {name} (20x20)...")
        
        for k in k_values:
            print(f"  Running RL Best-of-{k}...", end=" ", flush=True)
            rl_start_time = time.time()
            rl_ms, rl_sched = rollout_instance_with_k(inst, model, k_samples=k)
            rl_dur = time.time() - rl_start_time
            print(f" MS: {rl_ms} ({rl_dur:.2f}s)")

            for ratio in fix_ratios:
                print(f"    Hybrid CP (ratio={ratio})...", end=" ", flush=True)
                start = time.time()
                ms, _ = solve_with_rl_warmstart(inst['times'], inst['machines'], rl_sched, time_limit_s=30.0, fix_ratio=ratio)
                dur = time.time() - start
                print(f" MS: {ms} ({dur:.2f}s)")
                
                results.append({
                    "instance": name,
                    "k": k,
                    "fix_ratio": ratio,
                    "rl_ms_k": rl_ms,
                    "hybrid_ms": ms,
                    "hybrid_time": round(dur, 2)
                })

    df = pd.DataFrame(results)
    out_path = os.path.join(base_dir, "eval", "ablation_best_of_k_results.csv")
    df.to_csv(out_path, index=False)
    
    print("\nFinal Summary table (Avg Hybrid Makespan):")
    summary = df.groupby(["k", "fix_ratio"]).agg({"hybrid_ms": "mean", "hybrid_time": "mean"}).reset_index()
    print(summary.to_string())
    return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    ckpt = os.path.join(base_dir, "outputs", "models", "best_ppo_Phase_5.pt")
    if os.path.exists(ckpt):
        run_best_of_k_ablation(ckpt)
    else:
        print("Checkpoint not found.")
