import torch
from models.gin import HeteroGIN

def prepare_features(env, device):
    
    
    num_jobs = env.num_jobs
    num_machines = env.num_machines

    # Pull env tensors onto the model device (without mutating env)
    times = env.times.to(device)
    machines = env.machines.to(device)
    state = env.state.to(device)

    # --- Feature 1: processing time per op (flattened) ---
    processing_time = times.flatten().to(torch.float32)

    # --- Feature 2: remaining operations per job (flattened) ---
    remaining_ops_list = []
    for j in range(num_jobs):
        # sum of scheduled ops before each op
        prefix = state[j].cumsum(dim=0)  # not exactly used, but we’ll keep simple:
        for o in range(num_machines):
            scheduled_before = int(state[j, :o].sum().item())
            remaining_ops_list.append(num_machines - scheduled_before)
    remaining_ops = torch.tensor(remaining_ops_list, dtype=torch.float32, device=device)

    # --- Feature 3: machine load (vectorized) ---
    mask = (state == 0)  # unscheduled ops
    idx = machines[mask].flatten().to(device)                       # LongTensor
    vals = times[mask].to(torch.float32).flatten().to(device)       # FloatTensor
    machine_loads = torch.zeros(num_machines, device=device, dtype=torch.float32)
    if idx.numel() > 0:
        machine_loads.scatter_add_(0, idx, vals)
    machine_loads_rep = machine_loads[machines.flatten()].to(torch.float32)

    # --- Stack & normalize column-wise (still on dev) ---
    X = torch.stack([processing_time, remaining_ops, machine_loads_rep], dim=1)
    mean, std = X.mean(dim=0, keepdim=True), X.std(dim=0, keepdim=True) + 1e-6
    X = (X - mean) / std

    # --- Build edges on CPU then move to dev ---
    edge_index_dict_cpu = HeteroGIN.build_edge_index_dict(machines.detach().cpu())
    data = HeteroGIN.build_hetero_data(
        X,
        edge_index_dict_cpu[('op', 'job', 'op')].to(device),
        edge_index_dict_cpu[('op', 'machine', 'op')].to(device),
    )

    # dummy batch for pooling
    data['op'].batch = torch.zeros(X.size(0), dtype=torch.long, device=device)
    return data
