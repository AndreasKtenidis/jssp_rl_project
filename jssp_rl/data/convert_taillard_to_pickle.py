# jssp_rl/data/convert_taillard_to_pickle.py

import os
import pickle
from taillard_loader import load_taillard_instance

def convert_all_taillard(txt_folder):
    save_path = os.path.join("saved", "taillard_instances.pkl")
    instances = []
    for fname in sorted(os.listdir(txt_folder)):
        if fname.endswith(".txt"):
            print(f"➡️ Loading {fname}...")
            instance = load_taillard_instance(os.path.join(txt_folder, fname))
            print(f"✅ Finished {fname}")
            instances.append(instance)
            print(f"Loaded {fname}")
    
    os.makedirs("saved", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(instances, f)
    print(f"✅ Saved {len(instances)} Taillard instances to {save_path}")

if __name__ == "__main__":
    convert_all_taillard("C:/Users/andre/Desktop/Results JSSP/Taillard Benchmark instances/Ta_15x15") 
