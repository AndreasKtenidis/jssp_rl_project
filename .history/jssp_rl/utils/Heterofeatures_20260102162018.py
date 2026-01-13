import torch
from models.gin import HeteroGATv2

def prepare_features(env, device):
    """
    Build HeteroData features for HeteroGIN on the *given* device (model device).
    """
    num_jobs = env.num_jobs
    num_machines = env.num_machines

    times = env.times.to(device)
    machines = env.machines.to(device)
    state = env.state.to(device)

    # --- Feature 1: Processing time (flattened) ---
    processing_time = times.flatten().to(torch.float32)

    # --- Feature 2: Is scheduled (binary) ---
    is_scheduled = state.flatten().to(torch.float32)

    # --- Feature 3: Normalized op index (position in job) ---
    op_indices = torch.arange(num_machines, device=device).repeat(num_jobs)
    op_idx_norm = op_indices.float() / (num_machines - 1)

    # --- Feature 4: Is last op in job (binary) ---
    is_last_op = (op_indices == (num_machines - 1)).float()

    # --- Feature 5 & 6: min/max/gap processing time for op index (across jobs) ---
    times_cpu = times.detach().cpu()
    min_time_per_op = torch.tensor([
        times_cpu[:, o].min().item() for o in range(num_machines)
    ], dtype=torch.float32, device=device).repeat(num_jobs)
    max_time_per_op = torch.tensor([
        times_cpu[:, o].max().item() for o in range(num_machines)
    ], dtype=torch.float32, device=device).repeat(num_jobs)
    gap_max_min = max_time_per_op - min_time_per_op

        # --- Feature 7: Average remaining time for job ---
    remaining_time_per_job = []
    for j in range(num_jobs):
        job_state = state[j]
        job_times = times[j]
        unsched_mask = (job_state == 0)
        rem_ops = job_times[unsched_mask]
        if rem_ops.numel() > 0:
            rem_time = rem_ops.mean()
        else:
            rem_time = torch.tensor(0.0, device=device)
        remaining_time_per_job.append(rem_time)
    remaining_time_per_job = torch.stack(remaining_time_per_job)  # shape: [num_jobs]

    # Expand to all operations (job i → num_machines ops)
    avg_remaining_job_time = remaining_time_per_job.repeat_interleave(num_machines)

    # --- Feature 8: Estimated start time per operation ---
    est_start_times = []
    for j in range(num_jobs):
        for o in range(num_machines):
            if state[j, o] == 1:
                est_start_times.append(torch.tensor(0.0, device=device))  # already scheduled
                continue

            # Predecessor end time
            if o == 0:
                pred_end = 0.0
            else:
                if state[j, o - 1] == 1:
                    pred_end = env.job_completion_times[j, o - 1].item()
                else:
                    pred_end = float('inf')  # not schedulable yet

            # Machine availability
            m_id = int(machines[j, o].item())
            mach_free = env.machine_available_times[m_id].item()

            est_start = max(pred_end, mach_free)
            est_start = 0.0 if est_start == float('inf') else est_start
            est_start_times.append(torch.tensor(est_start, device=device))

    est_start_time = torch.stack(est_start_times)

        # --- Machine-level features: machine load & unscheduled op count ---
    mask_unsched = (state == 0)
    unsched_times = times[mask_unsched]
    unsched_machines = machines[mask_unsched]

    machine_loads = torch.zeros(num_machines, device=device, dtype=torch.float32)
    machine_op_counts = torch.zeros(num_machines, device=device, dtype=torch.float32)

    if unsched_times.numel() > 0:
        for m_id in range(num_machines):
            mask_m = (unsched_machines == m_id)
            machine_loads[m_id] = unsched_times[mask_m].sum()
            machine_op_counts[m_id] = mask_m.sum()

    # Replicate per operation
    machine_loads_rep = torch.zeros_like(machines, dtype=torch.float32)
    machine_counts_rep = torch.zeros_like(machines, dtype=torch.float32)
    for j in range(num_jobs):
        for o in range(num_machines):
            m_id = machines[j, o].item()
            machine_loads_rep[j, o] = machine_loads[m_id]
            machine_counts_rep[j, o] = machine_op_counts[m_id]

    machine_loads_flat = machine_loads_rep.flatten()
    machine_counts_flat = machine_counts_rep.flatten()

    # --- Job-level Features: f10, f11, f12 ---

    # f10: Job Progress Ratio
    progress_per_job = (state == 1).sum(dim=1).float() / num_machines
    job_progress_ratio = progress_per_job.repeat_interleave(num_machines)

    # f11: Job Remaining Time
    remaining_time = []
    for j in range(num_jobs):
        rem = times[j][state[j] == 0].sum()
        remaining_time.append(rem)
    job_remaining_tensor = torch.stack(remaining_time).to(device)
    job_remaining_time = job_remaining_tensor.repeat_interleave(num_machines)

    # f12: Job Slack Time = CLB - Remaining
    clb = env.estimate_clb()
    job_slack = clb - job_remaining_tensor
    job_slack_time = job_slack.repeat_interleave(num_machines)



    # --- Stack all features (13 columns total) ---
    X = torch.stack([
        processing_time,       # f0
        is_scheduled,          # f1
        op_idx_norm,           # f2
        is_last_op,            # f3
        min_time_per_op,       # f4
        gap_max_min,           # f5
        avg_remaining_job_time,  # f6
        est_start_time,      # f7
        machine_loads_flat,   #f8
        machine_counts_flat,   #f9
        job_progress_ratio,    #f10
        job_remaining_time,    #f11
        job_slack_time         #f12

    ], dim=1)


    # --- Normalize continuous features only (not binary ones) ---
    binary_mask = torch.tensor([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool, device=device)
    X_cont = X[:, ~binary_mask]
    mean, std = X_cont.mean(dim=0, keepdim=True), X_cont.std(dim=0, keepdim=True) + 1e-6
    X[:, ~binary_mask] = (X_cont - mean) / std

    # --- Build edge_index_dict then HeteroData ---
    edge_index_dict_cpu = HeteroGATv2.build_edge_index_dict(machines.detach().cpu())
    data = HeteroGATv2.build_hetero_data(
        X,
        edge_index_dict_cpu[('op', 'job', 'op')].to(device),
        edge_index_dict_cpu[('op', 'machine', 'op')].to(device),
    )

    # Dummy batch for pooling
    data['op'].batch = torch.zeros(X.size(0), dtype=torch.long, device=device)
    return data
