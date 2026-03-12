from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


RuleName = str
OpTuple = Tuple[int, int, int, int, int, int]


_RULE_ALIASES = {
    "mwkr": "mwkr",
    "mtwr": "mwkr",
    "most_work_remaining": "mwkr",
    "most_total_work_remaining": "mwkr",
    "spt": "spt",
    "shortest_processing_time": "spt",
    "fifo": "fifo",
    "first_in_first_out": "fifo",
}


def supported_rules() -> List[str]:
    return ["mwkr", "spt", "fifo"]


def _normalize_rule(rule: RuleName) -> RuleName:
    key = str(rule).strip().lower()
    if key not in _RULE_ALIASES:
        raise ValueError(f"Unknown rule '{rule}'. Supported: {supported_rules()}")
    return _RULE_ALIASES[key]


def _to_numpy_2d(x: Sequence[Sequence[int]]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}")
    return arr


def _remaining_work(times_np: np.ndarray, next_op: np.ndarray, job_id: int) -> int:
    op_idx = int(next_op[job_id])
    if op_idx >= times_np.shape[1]:
        return 0
    return int(np.sum(times_np[job_id, op_idx:]))


def _select_conflict_op(
    conflict_set: List[OpTuple],
    rule: RuleName,
    times_np: np.ndarray,
    next_op: np.ndarray,
    job_ready: np.ndarray,
) -> OpTuple:
    # tuple: (job_id, op_idx, machine, proc_time, est, ect)
    if rule == "mwkr":
        return min(
            conflict_set,
            key=lambda op: (
                -_remaining_work(times_np, next_op, op[0]),
                op[4],   # est
                op[3],   # proc time
                op[0],   # job id
            ),
        )

    if rule == "spt":
        return min(
            conflict_set,
            key=lambda op: (
                op[3],   # proc time
                op[4],   # est
                -_remaining_work(times_np, next_op, op[0]),
                op[0],   # job id
            ),
        )

    # FIFO
    return min(
        conflict_set,
        key=lambda op: (
            int(job_ready[op[0]]),  # arrival to machine queue
            op[4],                  # est
            op[0],                  # job id
        ),
    )


def generate_pdr_schedule(
    times: Sequence[Sequence[int]],
    machines: Sequence[Sequence[int]],
    rule: RuleName = "mwkr",
) -> Tuple[int, List[Dict[str, int]]]:
    """Construct a feasible schedule with GT + dispatching rule.

    Args:
        times: Processing times [num_jobs, num_machines].
        machines: Machine ids [num_jobs, num_machines], assumed 0-based.
        rule: One of "mwkr"/"mtwr", "spt", "fifo".

    Returns:
        makespan, schedule
    """
    rule = _normalize_rule(rule)
    times_np = _to_numpy_2d(times).astype(np.int64, copy=False)
    machines_np = _to_numpy_2d(machines).astype(np.int64, copy=False)

    if times_np.shape != machines_np.shape:
        raise ValueError(
            f"times and machines shapes differ: {times_np.shape} vs {machines_np.shape}"
        )

    num_jobs, num_ops = times_np.shape
    if num_jobs == 0 or num_ops == 0:
        return 0, []

    max_machine = int(np.max(machines_np))
    min_machine = int(np.min(machines_np))
    if min_machine < 0:
        raise ValueError("Machine ids must be non-negative")
    num_machines = max_machine + 1

    next_op = np.zeros(num_jobs, dtype=np.int64)
    job_ready = np.zeros(num_jobs, dtype=np.int64)
    machine_ready = np.zeros(num_machines, dtype=np.int64)
    schedule: List[Dict[str, int]] = []

    total_ops = num_jobs * num_ops

    for _ in range(total_ops):
        available: List[OpTuple] = []
        for j in range(num_jobs):
            o = int(next_op[j])
            if o >= num_ops:
                continue

            m = int(machines_np[j, o])
            p = int(times_np[j, o])
            est = int(max(job_ready[j], machine_ready[m]))
            ect = est + p
            available.append((j, o, m, p, est, ect))

        if not available:
            break

        # Giffler-Thompson pivot: operation with smallest earliest completion.
        pivot = min(available, key=lambda op: (op[5], op[4], op[0], op[1]))
        pivot_machine = pivot[2]
        pivot_ect = pivot[5]

        # Conflict set on pivot machine.
        conflict_set = [
            op for op in available if op[2] == pivot_machine and op[4] < pivot_ect
        ]
        chosen = _select_conflict_op(conflict_set, rule, times_np, next_op, job_ready)

        j, o, m, p, est, _ = chosen
        start = est
        end = start + p

        schedule.append(
            {
                "job_id": int(j),
                "operation_index": int(o),
                "machine": int(m),
                "start_time": int(start),
                "end_time": int(end),
            }
        )

        job_ready[j] = end
        machine_ready[m] = end
        next_op[j] += 1

    makespan = int(np.max(job_ready)) if len(schedule) == total_ops else int(1e18)
    schedule.sort(
        key=lambda e: (e["start_time"], e["machine"], e["job_id"], e["operation_index"])
    )
    return makespan, schedule
