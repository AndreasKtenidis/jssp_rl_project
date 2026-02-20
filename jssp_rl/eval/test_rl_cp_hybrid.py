"""test_rl_cp_hybrid.py - Hybrid RL+CP evaluation on standard benchmarks
========================================================================
Evaluates RL (greedy), CP (cold), and Hybrid (RL-warm-started CP) on all
standard JSSP benchmarks (FT, Taillard, DMU).

Results include BKS comparison and optimality gaps.
"""

import os
import json
import pickle
import torch
import time
import pandas as pd
from models.Hetero_actor_critic_ppo import ActorCriticPPO
from env.jssp_environment import JSSPEnvironment
from cp.cp_solver import solve_with_rl_warmstart, solve_instance_your_version
from utils.features import make_hetero_data
from config import TRAINED_SIZES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def get_rl_schedule(instance, model):
    times = instance['times'] if isinstance(instance['times'], torch.Tensor) else torch.tensor(instance['times'])
    machines = instance['machines'] if isinstance(instance['machines'], torch.Tensor) else torch.tensor(instance['machines'])
    times = times.to(device)
    machines = machines.to(device)
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
        if step_count % 100 == 0:
            print(".", end="", flush=True)

    print(" (RL Done)", flush=True)
    return env.get_makespan(), env.extract_job_assignments()


def get_time_limit_for_size(n_jobs, n_machines):
    """Adaptive time limit based on instance size."""
    total_ops = n_jobs * n_machines
    if total_ops <= 50:
        return 10       # FT06 etc.
    elif total_ops <= 200:
        return 20       # 10x10, 15x15
    elif total_ops <= 500:
        return 30       # 20x15, 20x20, 30x15
    elif total_ops <= 1000:
        return 60       # 30x20, 50x15, 50x20
    else:
        return 120      # 100x20


def run_hybrid_experiment(checkpoint_path, time_limit_override=None, fix_ratio=0.0,
                          benchmarks_to_run=None):
    """Run RL vs CP vs Hybrid on all benchmarks.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        time_limit_override: If set, overrides adaptive time limit
        fix_ratio: Fraction of RL solution to fix in hybrid mode
        benchmarks_to_run: List of benchmark names to evaluate (default: all)
    """
    print(f"\n{'='*60}")
    print(f"  HYBRID EXPERIMENT: RL vs CP vs RL+CP")
    print(f"{'='*60}")
    print(f"  Model: {os.path.basename(checkpoint_path)}")
    print(f"  Fix Ratio: {fix_ratio}")
    print(f"  Time Limit: {'adaptive' if time_limit_override is None else f'{time_limit_override}s'}")
    
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

    # 2. Load Benchmarks
    base_dir = os.path.dirname(os.path.dirname(__file__))
    saved_dir = os.path.join(base_dir, "saved")
    
    benchmark_files = {
        "FT": "benchmark_ft.pkl",
        "Taillard": "benchmark_taillard.pkl",
        "DMU": "benchmark_dmu.pkl",
    }
    
    if benchmarks_to_run is None:
        benchmarks_to_run = list(benchmark_files.keys())
    
    all_results = []
    
    for bm_name in benchmarks_to_run:
        filename = benchmark_files.get(bm_name)
        if not filename:
            continue
        path = os.path.join(saved_dir, filename)
        if not os.path.exists(path):
            print(f"[SKIP] {bm_name}: {filename} not found")
            continue
        
        with open(path, "rb") as f:
            instances = pickle.load(f)
        
        print(f"\n--- {bm_name}: {len(instances)} instances ---")
        
        for i, inst in enumerate(instances):
            name = inst.get("name", f"inst_{i}")
            times_t = inst['times'] if isinstance(inst['times'], torch.Tensor) else torch.tensor(inst['times'])
            machines_t = inst['machines'] if isinstance(inst['machines'], torch.Tensor) else torch.tensor(inst['machines'])
            N, M = times_t.shape
            bks = inst.get("bks", -1)
            dist_type = "In-Distribution" if (N, M) in TRAINED_SIZES else "Generalization"
            
            tl = time_limit_override if time_limit_override else get_time_limit_for_size(N, M)
            
            print(f"\n[{bm_name}] {name} ({N}x{M}) | BKS={bks} | TL={tl}s")
            
            # A. Pure RL (greedy)
            print("  [RL] Rolling out...", end=" ", flush=True)
            start_rl = time.time()
            rl_makespan, rl_sched = get_rl_schedule(inst, model)
            rl_time = time.time() - start_rl
            
            # B. Pure CP (Cold start)
            print(f"  [CP] Solving ({tl}s)...", end=" ", flush=True)
            start_cp = time.time()
            times_np = times_t.numpy()
            machines_np = machines_t.numpy()
            cp_cold_ms, _ = solve_instance_your_version(times_np, machines_np, time_limit_s=tl)
            cp_cold_time = time.time() - start_cp
            print(f"Done. MS={cp_cold_ms}")
            
            # C. Hybrid (Warm start)
            print(f"  [Hybrid] Solving ({tl}s)...", end=" ", flush=True)
            start_hybrid = time.time()
            cp_warm_ms, _ = solve_with_rl_warmstart(
                times_np, machines_np,
                rl_schedule=rl_sched,
                time_limit_s=tl,
                fix_ratio=fix_ratio
            )
            hybrid_time = time.time() - start_hybrid
            print(f"Done. MS={cp_warm_ms}")
            
            # Compute gaps vs BKS
            gap_rl = ((rl_makespan - bks) / bks * 100) if bks > 0 else -1
            gap_cp = ((cp_cold_ms - bks) / bks * 100) if bks > 0 and cp_cold_ms > 0 else -1
            gap_hybrid = ((cp_warm_ms - bks) / bks * 100) if bks > 0 and cp_warm_ms > 0 else -1
            
            best_method = min(
                [("RL", rl_makespan), ("CP", cp_cold_ms), ("Hybrid", cp_warm_ms)],
                key=lambda x: x[1] if x[1] > 0 else float('inf')
            )
            
            print(f"  -> RL={rl_makespan:.0f}({gap_rl:.1f}%) | CP={cp_cold_ms}({gap_cp:.1f}%) | "
                  f"Hybrid={cp_warm_ms}({gap_hybrid:.1f}%) | Best={best_method[0]}")
            
            all_results.append({
                "benchmark": bm_name,
                "instance": name,
                "size": f"{N}x{M}",
                "distribution": dist_type,
                "num_jobs": N,
                "num_machines": M,
                "bks": bks,
                "rl_ms": rl_makespan,
                "cp_cold_ms": cp_cold_ms,
                "hybrid_ms": cp_warm_ms,
                "gap_rl_pct": round(gap_rl, 2),
                "gap_cp_pct": round(gap_cp, 2),
                "gap_hybrid_pct": round(gap_hybrid, 2),
                "rl_time": round(rl_time, 2),
                "cp_cold_time": round(cp_cold_time, 2),
                "hybrid_time": round(hybrid_time, 2),
                "best_method": best_method[0],
                "time_limit": tl,
            })
    
    # Save results
    df = pd.DataFrame(all_results)
    eval_dir = os.path.join(base_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    out_csv = os.path.join(eval_dir, "hybrid_experiment_results.csv")
    df.to_csv(out_csv, index=False)
    
    # Print summary split by distribution
    for dist_label in ["In-Distribution", "Generalization"]:
        subset = df[df["distribution"] == dist_label]
        if subset.empty:
            continue
        print(f"\n{'='*60}")
        print(f"  HYBRID RESULTS: {dist_label.upper()}")
        print(f"{'='*60}")
        
        summary = subset.groupby(["benchmark", "size"]).agg({
            "bks": "mean",
            "rl_ms": "mean",
            "cp_cold_ms": "mean",
            "hybrid_ms": "mean",
            "gap_rl_pct": "mean",
            "gap_cp_pct": "mean",
            "gap_hybrid_pct": "mean",
        }).round(1)
        
        print(summary.to_string())
    
    print(f"\n[Finished] Results saved to {out_csv}")
    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Find best checkpoint
    model_dir = os.path.join(base_dir, "outputs", "models")
    ckpt = None
    for candidate in ["best_ppo_latest.pt", "best_ppo_Phase_7.pt", "best_ppo_Phase_6.pt",
                       "best_ppo_Phase_5.pt", "best_ppo_Phase_4.pt", "best_ppo_Phase_3.pt"]:
        path = os.path.join(model_dir, candidate)
        if os.path.exists(path):
            ckpt = path
            break

    if ckpt and os.path.exists(ckpt):
        run_hybrid_experiment(ckpt, fix_ratio=0.4)
    else:
        print("No model checkpoint found. Run training first.")
