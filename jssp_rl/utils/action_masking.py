import torch

def select_top_k_actions(env, top_k=5, device=None):
    
    num_ops = env.num_jobs * env.num_machines
    valid_mask = torch.zeros(num_ops, dtype=torch.bool, device=device)
    priorities = []

    for job_id in range(env.num_jobs):
        for op_idx in range(env.num_machines):
            idx = job_id * env.num_machines + op_idx
            if env.state[job_id, op_idx] == 0 and (op_idx == 0 or env.state[job_id, op_idx - 1] == 1):
                est = env.job_completion_times[job_id, op_idx - 1] if op_idx > 0 else 0
                pt = env.times[job_id, op_idx]

                valid_mask[idx] = 1
                priorities.append((est, -pt, idx))

    if len(priorities) == 0:
        top_k_mask = torch.zeros(num_ops, dtype=torch.bool, device=device)
        return torch.tensor([], dtype=torch.long, device=device), top_k_mask

    priorities.sort(key=lambda x: (x[0], x[1]))
    selected_indices = [idx for *_, idx in priorities[:min(top_k, len(priorities))]]

    top_k_mask = torch.zeros_like(valid_mask)
    top_k_mask[selected_indices] = 1

    return torch.tensor(selected_indices, dtype=torch.long, device=device), top_k_mask
