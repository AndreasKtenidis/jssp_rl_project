import torch
import pickle
import random
import os
import numpy as np

def generate_random_jssp(num_jobs=15, num_machines=15):
    machines_list = []
    times_list = []
    for _ in range(num_jobs):
        mach_order = list(range(num_machines))
        random.shuffle(mach_order)
        durations = [random.randint(1, 99) for _ in range(num_machines)]
        machines_list.append(mach_order)
        times_list.append(durations)
    return {
        "times": torch.tensor(times_list, dtype=torch.float32),
        "machines": torch.tensor(machines_list, dtype=torch.long)
    }

def main():
    sizes = [
        (6, 6), (10, 10), 
        (20, 15), (20, 20), 
        (30, 15), (30, 20), 
        (50, 15), (50, 20)
    ]
    save_dir = "saved"
    os.makedirs(save_dir, exist_ok=True)
    
    for num_jobs, num_machines in sizes:
        filename = f"Synthetic_instances_{num_jobs}x{num_machines}_5000.pkl"
        full_path = os.path.join(save_dir, filename)
        
        if os.path.exists(full_path):
            print(f"Skipping {num_jobs}x{num_machines}, already exists.")
            continue

        print(f"Generating {num_jobs}x{num_machines} (500 instances)...")
        dataset = [generate_random_jssp(num_jobs, num_machines) for _ in range(500)]
        with open(full_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Saved to {full_path}")

if __name__ == "__main__":
    main()
