import torch


from torch_geometric.data import HeteroData
from models.gin import HeteroGATv2

def prepare_features(env, device):
    def normalize(tensor):
        tensor = tensor.to(torch.float32).to(device)
        return (tensor - tensor.mean()) / (tensor.std() + 1e-6)

    num_jobs = env.num_jobs
    num_machines = env.num_machines
    num_nodes = num_jobs * num_machines

    processing_time = env.times.flatten()


    remaining_ops = []
    for job_id in range(num_jobs):
        for op_index in range(num_machines):
            scheduled = torch.sum(env.state[job_id, :op_index]).item()
            remaining = num_machines - scheduled
            remaining_ops.append(remaining)
    remaining_ops = torch.tensor(remaining_ops, dtype=torch.float32, device=device)

    machine_loads = torch.zeros(num_machines, device=device)
    for job_id in range(num_jobs):
        for op_index in range(num_machines):
            machine_id = env.machines[job_id, op_index].item()
            if env.state[job_id, op_index] == 0:
                machine_loads[machine_id] += env.times[job_id, op_index].item()
    machine_loads_rep = machine_loads[env.machines.flatten()].to(torch.float32)

    x = torch.stack([
        normalize(processing_time),
        normalize(remaining_ops),
        normalize(machine_loads_rep),
    ], dim=1).to(device)

    return x


def make_hetero_data(env, device):
    x = prepare_features(env, device)
    edge_index_dict = HeteroGATv2.build_edge_index_dict(env.machines)
    
    data = HeteroData()
    data['op'].x = x
    
    # Move edge indices to device
    # Note: build_edge_index_dict returns on CPU by default
    job_edges = edge_index_dict[('op', 'job', 'op')].to(device)
    mach_edges = edge_index_dict[('op', 'machine', 'op')].to(device)
    
    data['op', 'job', 'op'].edge_index = job_edges
    data['op', 'machine', 'op'].edge_index = mach_edges
    
    return data
