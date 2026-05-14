import torch
from torch_geometric.data import HeteroData


def _normalize_continuous_columns(x, binary_columns=None):
    if x.numel() == 0:
        return x

    if binary_columns is None:
        binary_columns = torch.zeros(x.size(1), dtype=torch.bool, device=x.device)

    continuous = ~binary_columns
    if continuous.any():
        values = x[:, continuous]
        mean = values.mean(dim=0, keepdim=True)
        std = values.std(dim=0, keepdim=True) + 1e-6
        x[:, continuous] = (values - mean) / std
    return x


def _operation_features(env, device):
    ready_ops = set()
    for action_id in env.get_available_actions():
        op_id, _, _ = env.decode_action(action_id)
        ready_ops.add(op_id)

    features = []
    for op_id, (job_id, op_idx) in enumerate(env.op_to_job_op):
        job = env.instance[job_id]
        operation = job[op_idx]

        is_scheduled = float(env.state[op_id].item())
        is_ready = float(op_id in ready_ops)
        op_idx_norm = op_idx / max(1, len(job) - 1)
        is_last_op = float(op_idx == len(job) - 1)

        option_durations = [duration for _, duration in operation]
        min_duration = float(min(option_durations))
        mean_duration = float(sum(option_durations) / len(option_durations))
        option_count = float(len(option_durations))

        remaining_ops = 0.0
        remaining_work_min = 0.0
        for later_idx in range(op_idx, len(job)):
            later_op_id = env.job_op_to_op[(job_id, later_idx)]
            if env.state[later_op_id] == 0:
                remaining_ops += 1.0
                remaining_work_min += min(duration for _, duration in job[later_idx])

        if op_idx == 0:
            predecessor_end = 0.0
        else:
            prev_op_id = env.job_op_to_op[(job_id, op_idx - 1)]
            predecessor_end = float(env.op_completion_times[prev_op_id].item())

        features.append([
            is_scheduled,
            is_ready,
            op_idx_norm,
            is_last_op,
            min_duration,
            mean_duration,
            option_count,
            remaining_ops,
            remaining_work_min,
            predecessor_end,
        ])

    x = torch.tensor(features, dtype=torch.float32, device=device)
    binary_columns = torch.tensor(
        [True, True, False, True, False, False, False, False, False, False],
        dtype=torch.bool,
        device=device,
    )
    return _normalize_continuous_columns(x, binary_columns=binary_columns)


def _machine_features(env, device):
    remaining_load = torch.zeros(env.num_machines, dtype=torch.float32, device=device)
    remaining_count = torch.zeros(env.num_machines, dtype=torch.float32, device=device)

    for action_id, (op_id, machine_id, duration) in enumerate(env.action_to_op_machine_duration):
        if env.state[op_id] == 0:
            remaining_load[machine_id] += float(duration)
            remaining_count[machine_id] += 1.0

    available_time = env.machine_available_times.to(device=device, dtype=torch.float32)
    x = torch.stack([available_time, remaining_load, remaining_count], dim=1)
    return _normalize_continuous_columns(x)


def _edge_data(env, device):
    precedence_edges = []
    for job_id, job in enumerate(env.instance):
        for op_idx in range(len(job) - 1):
            src = env.job_op_to_op[(job_id, op_idx)]
            dst = env.job_op_to_op[(job_id, op_idx + 1)]
            precedence_edges.append([src, dst])

    eligible_edges = []
    durations = []
    action_ids = []
    for action_id, (op_id, machine_id, duration) in enumerate(env.action_to_op_machine_duration):
        eligible_edges.append([op_id, machine_id])
        durations.append([float(duration)])
        action_ids.append(action_id)

    if precedence_edges:
        precedence_edge_index = torch.tensor(
            precedence_edges, dtype=torch.long, device=device
        ).t().contiguous()
    else:
        precedence_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    if eligible_edges:
        op_machine_edge_index = torch.tensor(
            eligible_edges, dtype=torch.long, device=device
        ).t().contiguous()
        machine_op_edge_index = op_machine_edge_index.flip(0)
        edge_attr = torch.tensor(durations, dtype=torch.float32, device=device)
        edge_attr = _normalize_continuous_columns(edge_attr)
        action_index = torch.tensor(action_ids, dtype=torch.long, device=device)
    else:
        op_machine_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        machine_op_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 1), dtype=torch.float32, device=device)
        action_index = torch.empty((0,), dtype=torch.long, device=device)

    return precedence_edge_index, op_machine_edge_index, machine_op_edge_index, edge_attr, action_index


def prepare_features(env, device):
    return {
        "op": _operation_features(env, device),
        "machine": _machine_features(env, device),
    }


def make_hetero_data(env, device):
    data = HeteroData()
    features = prepare_features(env, device)
    data["op"].x = features["op"]
    data["machine"].x = features["machine"]

    (
        precedence_edge_index,
        op_machine_edge_index,
        machine_op_edge_index,
        edge_attr,
        action_index,
    ) = _edge_data(env, device)

    data["op", "precedes", "op"].edge_index = precedence_edge_index
    data["op", "eligible", "machine"].edge_index = op_machine_edge_index
    data["op", "eligible", "machine"].edge_attr = edge_attr
    data["op", "eligible", "machine"].action_index = action_index
    data["machine", "processes", "op"].edge_index = machine_op_edge_index
    data["machine", "processes", "op"].edge_attr = edge_attr
    data["machine", "processes", "op"].action_index = action_index

    action_tuples = env.action_to_op_machine_duration
    data.action_op_ids = torch.tensor([item[0] for item in action_tuples], dtype=torch.long, device=device)
    data.action_machine_ids = torch.tensor([item[1] for item in action_tuples], dtype=torch.long, device=device)
    data.action_durations = torch.tensor([item[2] for item in action_tuples], dtype=torch.float32, device=device)

    return data
