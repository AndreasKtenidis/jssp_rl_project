import os
import pickle
import pandas as pd
import time
from cp.cp_solver import solve_instance_your_version

def run_pure_cp_baseline(num_instances=5, time_limit=30.0):
    print(f"Starting Pure CP Baseline (Time Limit: {time_limit}s)")
    
    # Load Instances
    base_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base_dir, "saved", "benchmark_taillard.pkl")
    with open(path, "rb") as f:
        instances = pickle.load(f)[:num_instances]

    results = []

    for inst in instances:
        name = inst['name']
        print(f"Evaluating {name} with Pure CP...")
        times_np = inst['times'].numpy() if hasattr(inst['times'], "numpy") else inst['times']
        machines_np = inst['machines'].numpy() if hasattr(inst['machines'], "numpy") else inst['machines']

        start = time.time()
        ms, _ = solve_instance_your_version(times_np, machines_np, time_limit_s=time_limit)
        dur = time.time() - start
        
        results.append({
            "instance": name,
            "makespan": ms,
            "time": round(dur, 2)
        })

    df = pd.DataFrame(results)
    out_path = os.path.join(base_dir, "eval", "pure_cp_baseline_results.csv")
    df.to_csv(out_path, index=False)
    
    print("\nPure CP Summary:")
    print(df.agg({"makespan": "mean", "time": "mean"}).to_string())
    return df

if __name__ == "__main__":
    run_pure_cp_baseline()
