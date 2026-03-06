import torch


from torch_geometric.data import HeteroData
from models.gin import HeteroGATv2

def prepare_features(env, device):
    def normalize(tensor):
        tensor = tensor.to(torch.float32).to(device)
        return (tensor - tensor.mean()) / (tensor.std() + 1e-6)

    num_jobs = env.num_jobs
    num_machines = env.num_machines

    # 1. Processing Time
    processing_time = env.times.flatten()

    # 2. Remaining Ops (Vectorized)
    # env.state is [Jobs, Machines] (1 if scheduled, 0 otherwise)
    scheduled_cumsum = torch.cumsum(env.state, dim=1)
    # Exclusive sum: how many are scheduled strictly BEFORE this op in the job
    scheduled_before = torch.cat([torch.zeros(num_jobs, 1, device=device), scheduled_cumsum[:, :-1]], dim=1)
    remaining_ops = (num_machines - scheduled_before).flatten()

    # 3. Machine Loads (Vectorized)
    # Sum of processing times of UNSCHEDULED operations for each machine
    unscheduled = (env.state == 0).float()
    masked_times = env.times * unscheduled
    
    machine_loads = torch.zeros(num_machines, device=device)
    machine_loads.scatter_add_(0, env.machines.flatten(), masked_times.flatten())
    machine_loads_rep = machine_loads[env.machines.flatten()].to(torch.float32)

    x = torch.stack([
        normalize(processing_time),
        normalize(remaining_ops),
        normalize(machine_loads_rep),
    ], dim=1).to(device)

    return x


def make_hetero_data(env, device, precomputed_edge_index=None):
    x = prepare_features(env, device)
    
    if precomputed_edge_index is None:
        edge_index_dict = HeteroGATv2.build_edge_index_dict(env.machines)
    else:
        edge_index_dict = precomputed_edge_index
    
    data = HeteroData()
    data['op'].x = x
    
    # Move edge indices to device (if not already there)
    job_edges = edge_index_dict[('op', 'job', 'op')].to(device)
    mach_edges = edge_index_dict[('op', 'machine', 'op')].to(device)
    
    data['op', 'job', 'op'].edge_index = job_edges
    data['op', 'machine', 'op'].edge_index = mach_edges
    
    return data
