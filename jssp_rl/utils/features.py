import torch

def prepare_features(env, edge_index):
    def normalize(tensor):
        tensor = tensor.to(torch.float32)
        return (tensor - tensor.mean()) / (tensor.std() + 1e-6)

    num_jobs = env.num_jobs
    num_machines = env.num_machines
    num_nodes = num_jobs * num_machines

    processing_time = torch.tensor(env.times, dtype=torch.float32).flatten()

    remaining_ops = []
    for job_id in range(num_jobs):
        for op_index in range(num_machines):
            scheduled = torch.sum(env.state[job_id, :op_index]).item()
            remaining = num_machines - scheduled
            remaining_ops.append(remaining)
    remaining_ops = torch.tensor(remaining_ops, dtype=torch.float32)

    machine_loads = torch.zeros(num_machines)
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
    ], dim=1)

    return x
