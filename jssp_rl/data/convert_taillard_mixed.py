import os
import torch
import pickle
import glob

def parse_taillard_file(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Header: num_jobs num_machines
    parts = lines[0].split()
    num_jobs, num_machines = int(parts[0]), int(parts[1])
    
    times = []
    machines = []
    
    # Find "Times" and "Machines" indices
    times_idx = -1
    mach_idx = -1
    for i, line in enumerate(lines):
        if line.lower() == "times":
            times_idx = i
        if line.lower() == "machines":
            mach_idx = i
            
    # Read times
    for i in range(times_idx + 1, times_idx + 1 + num_jobs):
        times.append([float(x) for x in lines[i].split()])
        
    # Read machines (convert 1-based to 0-based)
    for i in range(mach_idx + 1, mach_idx + 1 + num_jobs):
        machines.append([int(x) - 1 for x in lines[i].split()])
        
    return {
        "times": torch.tensor(times, dtype=torch.float32),
        "machines": torch.tensor(machines, dtype=torch.long),
        "name": os.path.basename(file_path)
    }

def main():
    target_dir = "/home/aktenidis/JSSPprojects/Taillard Benchmark instances/Ta_mixed"
    output_file = "/home/aktenidis/JSSPprojects/jssp_rl_project/jssp_rl/saved/taillard_mixed_instances.pkl"
    
    files = glob.glob(os.path.join(target_dir, "*.txt"))
    instances = []
    for f in sorted(files):
        print(f"Parsing {f}...")
        try:
            instances.append(parse_taillard_file(f))
        except Exception as e:
            print(f"Failed to parse {f}: {e}")
            
    with open(output_file, "wb") as f:
        pickle.dump(instances, f)
    print(f"Saved {len(instances)} instances to {output_file}")

if __name__ == "__main__":
    main()
