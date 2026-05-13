import torch
import pickle
import random
import os
import numpy as np

def generate_random_jssp(num_jobs, num_machines):
    machines_list = []
    times_list = []
    for _ in range(num_jobs):
        mach_order = list(range(num_machines))
        random.shuffle(mach_order)
        durations = [random.randint(1, 99) for _ in range(num_machines)]
        machines_list.append(mach_order)
        times_list.append(durations)
    return {
        "name": f"dt_{num_jobs}x{num_machines}",
        "times": torch.tensor(times_list, dtype=torch.float32),
        "machines": torch.tensor(machines_list, dtype=torch.long),
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "bks": -1  # Synthetic benchmarks usually don't have known BKS for these sizes
    }

def main():
    sizes = [
        (10, 10), (10, 100), (10, 1000),
        (100, 10), (100, 100), (100, 1000),
        (1000, 10), (1000, 100), (1000, 1000)
    ]
    save_dir = "saved"
    os.makedirs(save_dir, exist_ok=True)
    
    num_instances_per_size = 5  # Reasonable count for benchmarks
    
    all_instances = []
    
    for nj, nm in sizes:
        filename = f"DacolTeppan2022_{nj}_{nm}.pkl"
        full_path = os.path.join(save_dir, filename)
        
        print(f"Generating {nj}x{nm} ({num_instances_per_size} instances)...")
        dataset = []
        for i in range(num_instances_per_size):
            inst = generate_random_jssp(nj, nm)
            inst["name"] = f"dt_{nj}x{nm}_{i}"
            dataset.append(inst)
            
        with open(full_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Saved to {full_path}")
        
        all_instances.extend(dataset)

    # Combined file
    combined_path = os.path.join(save_dir, "benchmark_DacolTeppan2022.pkl")
    with open(combined_path, "wb") as f:
        pickle.dump(all_instances, f)
    print(f"Saved combined benchmark to {combined_path}")

if __name__ == "__main__":
    main()
