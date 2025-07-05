import os
import pickle
import pandas as pd
from cp_solver import run_cp_on_all



if __name__ == "__main__":
    # === Load dataset dynamically ===
    base_dir = os.path.dirname(os.path.dirname(__file__))  # jssp_rl/
    data_path = os.path.join(base_dir, "saved", "taillard_instances.pkl")

    with open(data_path, "rb") as f:
        instances = pickle.load(f)

    # === Run CP and save CSV ===
    results = run_cp_on_all(instances, save_gantt_dir=os.path.join(base_dir, "eval"))
    df = pd.DataFrame(results)
    df.to_csv("cp_makespans.csv", index=False)
    print("✅ CP makespans saved to cp/cp_makespans.csv")