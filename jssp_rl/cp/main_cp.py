# jssp_rl/cp/main_cp.py

import os
import pickle
import pandas as pd
from cp.cp_solver import run_cp_on_all

def run_cp_on_taillard():
    # === Load dataset dynamically ===
    base_dir = os.path.dirname(os.path.dirname(__file__))  # jssp_rl/
    data_path = os.path.join(base_dir, "saved", "taillard_instances.pkl")

    with open(data_path, "rb") as f:
        instances = pickle.load(f)

    # === Run CP and save CSV ===
    results = run_cp_on_all(instances, save_gantt_dir=os.path.join(base_dir, "eval","gantt_cp"))
    df = pd.DataFrame(results)

    save_path = os.path.join(base_dir, "cp", "cp_makespans.csv")
    df.to_csv(save_path, index=False)
    print(f"✅ CP makespans saved to {save_path}")

# Optional manual entry point
if __name__ == "__main__":
    run_cp_on_taillard()
