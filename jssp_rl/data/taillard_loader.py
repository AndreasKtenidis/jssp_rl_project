# jssp_rl/data/taillard_loader.py
import numpy as np

def load_taillard_instance(file_path):
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    num_jobs, num_machines = map(int, lines[0].split())

    # Locate "Times" and "Machines" blocks
    times_start = lines.index("Times") + 1
    machines_start = lines.index("Machines") + 1

    times_lines = lines[times_start:machines_start - 1]
    machines_lines = lines[machines_start:]

    # Parse times
    times = np.array([list(map(int, line.split())) for line in times_lines])
    # Parse machines (convert from 1-based to 0-based indexing)
    machines = np.array([list(map(lambda x: int(x) - 1, line.split())) for line in machines_lines])

    assert times.shape == (num_jobs, num_machines), f"Times shape mismatch: {times.shape}"
    assert machines.shape == (num_jobs, num_machines), f"Machines shape mismatch: {machines.shape}"

    return {"times": times, "machines": machines}