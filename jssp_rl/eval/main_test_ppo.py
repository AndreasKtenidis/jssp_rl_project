"""main_test_ppo.py - Evaluate PPO/Reptile models on ALL standard benchmarks
===========================================================================
Loads checkpoints and evaluates on:
  - Fisher-Thompson (FT06, FT10, FT20)
  - Taillard (Ta01-Ta80)
  - Demirkol (DMU01-DMU80)

Reports: Greedy makespan, Best-of-K makespan, BKS, Gap(%)
Saves results as CSV per benchmark + combined markdown summary.
"""

import os
import json
import pickle
import torch
import pandas as pd
from data.dataset import JSSPDataset
from models.Hetero_actor_critic_ppo import ActorCriticPPO
from env.jssp_environment import JSSPEnvironment
from utils.logging_utils import plot_gantt_chart
from torch_geometric.data import Data
from utils.features import make_hetero_data

from config import BEST_OF_K, SAVE_GANTT_FOR_BEST, TRAINED_SIZES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No PPO checkpoint found at {path}. Train the model first.")
    state = torch.load(path, map_location=device)
    return state


@torch.no_grad()
def rollout_instance_once(times, machines, model, stochastic=False):
    """Single rollout. If stochastic=True, sample from Categorical(logits)."""
    env = JSSPEnvironment(times, machines, device=device)
    env.reset()
    max_steps = env.num_jobs * env.num_machines + 50
    steps = 0

    while True:
        steps += 1
        if steps > max_steps:
            return float("inf"), []

        data = make_hetero_data(env, device)
        logits, _ = model(data)

        avail = env.get_available_actions()
        if not avail:
            return float("inf"), []

        mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=device)
        mask[avail] = True
        logits = logits.masked_fill(~mask, -1e10)

        if stochastic:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
        else:
            action = torch.argmax(logits).item()

        _, _, done, _ = env.step(action)
        if done:
            return env.get_makespan(), env.extract_job_assignments()


@torch.no_grad()
def run_rl_on_instance(times, machines, model, best_of_k=1):
    """Run greedy once, and optionally best-of-K with stochastic sampling."""
    greedy_ms, greedy_sched = rollout_instance_once(times, machines, model, stochastic=False)

    if best_of_k <= 1:
        return greedy_ms, greedy_sched, greedy_ms, greedy_sched

    best_makespan = greedy_ms
    best_schedule = greedy_sched
    for _ in range(best_of_k):
        makespan_k, schedule_k = rollout_instance_once(times, machines, model, stochastic=True)
        if makespan_k < best_makespan:
            best_makespan = makespan_k
            best_schedule = schedule_k

    return greedy_ms, greedy_sched, best_makespan, best_schedule


def load_benchmark_instances(saved_dir):
    """Load all benchmark pickle files and return a dict of benchmark_name -> instances."""
    benchmarks = {}
    
    for bm_name, filename in [
        ("FT", "benchmark_ft.pkl"),
        ("Taillard", "benchmark_taillard.pkl"),
        ("DMU", "benchmark_dmu.pkl"),
    ]:
        path = os.path.join(saved_dir, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                instances = pickle.load(f)
            benchmarks[bm_name] = instances
            print(f"  [OK] {bm_name}: {len(instances)} instances loaded")
        else:
            print(f"  [SKIP] {bm_name}: {filename} not found")
    
    # Fallback: load old taillard_mixed if no new benchmarks found
    if not benchmarks:
        old_path = os.path.join(saved_dir, "taillard_mixed_instances.pkl")
        if os.path.exists(old_path):
            with open(old_path, "rb") as f:
                instances = pickle.load(f)
            benchmarks["Taillard_Legacy"] = instances
            print(f"  [Fallback] Loaded legacy taillard_mixed_instances.pkl: {len(instances)} instances")
    
    return benchmarks


def evaluate_on_benchmark(model, instances, benchmark_name, eval_dir, best_of_k=10, save_gantt=False):
    """Evaluate model on a set of benchmark instances."""
    results = []
    gantt_dir = os.path.join(eval_dir, f"gantt_{benchmark_name.lower()}")
    if save_gantt:
        os.makedirs(gantt_dir, exist_ok=True)
    
    for idx, inst in enumerate(instances):
        name = inst.get("name", f"inst_{idx}")
        times = inst["times"] if isinstance(inst["times"], torch.Tensor) else torch.tensor(inst["times"], dtype=torch.float32)
        machines = inst["machines"] if isinstance(inst["machines"], torch.Tensor) else torch.tensor(inst["machines"], dtype=torch.long)
        N, M = times.shape
        bks = inst.get("bks", -1)
        dist_type = "In-Distribution" if (N, M) in TRAINED_SIZES else "Generalization"
        
        print(f"  [{benchmark_name}] {name} ({N}x{M})...", end=" ", flush=True)
        
        greedy_ms, greedy_sched, best_ms, best_sched = run_rl_on_instance(
            times, machines, model, best_of_k=best_of_k
        )
        
        # Calculate gap vs BKS
        gap_greedy = ((greedy_ms - bks) / bks * 100) if bks > 0 else -1
        gap_best = ((best_ms - bks) / bks * 100) if bks > 0 else -1
        
        print(f"Greedy={greedy_ms:.0f} Best-of-{best_of_k}={best_ms:.0f} BKS={bks} Gap={gap_best:.1f}%")
        
        results.append({
            "benchmark": benchmark_name,
            "instance": name,
            "size": f"{N}x{M}",
            "distribution": dist_type,
            "num_jobs": N,
            "num_machines": M,
            "bks": bks,
            "rl_greedy": greedy_ms,
            f"rl_best_of_{best_of_k}": best_ms,
            "gap_greedy_pct": round(gap_greedy, 2),
            f"gap_best_of_{best_of_k}_pct": round(gap_best, 2),
        })
        
        if save_gantt and best_sched:
            gantt_path = os.path.join(gantt_dir, f"gantt_{name}.png")
            try:
                plot_gantt_chart(best_sched, save_path=gantt_path, show_op_index=True)
            except Exception:
                pass
    
    return pd.DataFrame(results)


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    saved_dir = os.path.join(base_dir, "saved")
    eval_dir = os.path.join(base_dir, "eval")
    model_dir = os.path.join(base_dir, "outputs", "models")
    os.makedirs(eval_dir, exist_ok=True)

    # Load all benchmarks
    print(f"\n=== Loading Benchmark Instances ===")
    benchmarks = load_benchmark_instances(saved_dir)
    
    if not benchmarks:
        print("[Error] No benchmark files found. Run `python data/download_benchmarks.py` first.")
        return

    # Find best checkpoint
    ckpt_path = None
    for candidate in ["best_ppo_latest.pt", "best_ppo_Phase_7.pt", "best_ppo_Phase_6.pt",
                       "best_ppo_Phase_5.pt", "best_ppo_Phase_4.pt", "best_ppo_Phase_3.pt"]:
        path = os.path.join(model_dir, candidate)
        if os.path.exists(path):
            ckpt_path = path
            break
    
    if ckpt_path is None:
        print("[Error] No model checkpoint found. Train the model first.")
        return
    
    print(f"\n=== Loading Model: {os.path.basename(ckpt_path)} ===")
    model = ActorCriticPPO(
        node_input_dim=3,
        gnn_hidden_dim=64,
        gnn_output_dim=32,
        actor_hidden_dim=32,
        critic_hidden_dim=32
    ).to(device)
    
    state_dict = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    
    # Evaluate on each benchmark
    all_results = []
    
    for bm_name, instances in benchmarks.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating on {bm_name} ({len(instances)} instances)")
        print(f"{'='*60}")
        
        df = evaluate_on_benchmark(
            model, instances, bm_name, eval_dir,
            best_of_k=BEST_OF_K,
            save_gantt=SAVE_GANTT_FOR_BEST
        )
        
        # Save per-benchmark CSV
        csv_path = os.path.join(eval_dir, f"benchmark_{bm_name.lower()}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"  -> Saved to {csv_path}")
        
        all_results.append(df)
    
    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)
    combined_path = os.path.join(eval_dir, "benchmark_all_results.csv")
    combined.to_csv(combined_path, index=False)
    
    # Print summary table
    bok_col = f"rl_best_of_{BEST_OF_K}"
    gap_col = f"gap_best_of_{BEST_OF_K}_pct"
    
    for dist_label in ["In-Distribution", "Generalization"]:
        subset = combined[combined["distribution"] == dist_label]
        if subset.empty:
            continue
        print(f"\n{'='*60}")
        print(f"  {dist_label.upper()} RESULTS")
        print(f"{'='*60}")
        
        summary = subset.groupby(["benchmark", "size"]).agg({
            "bks": "mean",
            "rl_greedy": "mean",
            bok_col: "mean",
            "gap_greedy_pct": "mean",
            gap_col: "mean",
        }).round(1)
        
        print(summary.to_string())
    
    print(f"\nSaved combined results to {combined_path}")


if __name__ == "__main__":
    main()
