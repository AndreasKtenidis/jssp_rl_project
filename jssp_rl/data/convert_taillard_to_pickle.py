# jssp_rl/data/convert_taillard_to_pickle.py
import os
import pickle
import torch
from taillard_loader import load_taillard_instance

def convert_all_taillard(txt_folder):
    save_path = os.path.join("saved", "Synthetic_instances_15x15_505.pkl")
    instances = []

    for fname in sorted(os.listdir(txt_folder)):
        if fname.endswith(".txt"):
            print(f"➡️ Loading {fname}...")
            instance = load_taillard_instance(os.path.join(txt_folder, fname))

            # ✅ Convert to torch.Tensor directly
            instance["times"] = torch.tensor(instance["times"], dtype=torch.float32)
            instance["machines"] = torch.tensor(instance["machines"], dtype=torch.long)

            instances.append(instance)
            print(f"✅ Finished {fname}")

    os.makedirs("saved", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(instances, f)

    print(f"✅ Saved {len(instances)} Taillard instances to {save_path}")

if __name__ == "__main__":
    convert_all_taillard("C:/Users/andre/Desktop/Results JSSP/Synthetic_data/Synthetic_500/Synthetic_15x15")