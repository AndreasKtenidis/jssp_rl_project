"""Converters between classic JSSP data and the unified FJSSP format."""

from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected in this project.
    torch = None

from .common import Instance, Job, MachineOption, validate_instance

ClassicJob = Sequence[Tuple[int, int]]
ClassicInstance = Sequence[ClassicJob]


def _to_python_matrix(values: Any) -> List[List[Any]]:
    """Convert tensor/array/list-like 2D data to a nested Python list."""
    if torch is not None and torch.is_tensor(values):
        return values.detach().cpu().tolist()
    if hasattr(values, "tolist"):
        return values.tolist()
    return [list(row) for row in values]


def jssp_to_fjssp(jobs: ClassicInstance) -> Instance:
    """Convert classic list-style JSSP jobs to unified FJSSP format.

    Example:
        [(0, 5), (1, 3)] -> [[(0, 5)], [(1, 3)]]

    A full instance is represented as a list of jobs, where each job is a list
    of ``(machine_id, processing_time)`` pairs.
    """
    instance: Instance = []
    for job in jobs:
        unified_job: Job = []
        for machine_id, processing_time in job:
            unified_job.append([(int(machine_id), int(processing_time))])
        instance.append(unified_job)
    validate_instance(instance)
    return instance


def tensor_jssp_to_fjssp(times: Any, machines: Any) -> Instance:
    """Convert classic JSSP times/machines matrices to unified FJSSP format."""
    times_matrix = _to_python_matrix(times)
    machines_matrix = _to_python_matrix(machines)

    if len(times_matrix) != len(machines_matrix):
        raise ValueError(
            f"times and machines must have the same number of jobs: "
            f"{len(times_matrix)} vs {len(machines_matrix)}"
        )

    instance: Instance = []
    for job_id, (time_row, machine_row) in enumerate(zip(times_matrix, machines_matrix)):
        if len(time_row) != len(machine_row):
            raise ValueError(
                f"times and machines row {job_id} must have the same length: "
                f"{len(time_row)} vs {len(machine_row)}"
            )
        job: Job = []
        for processing_time, machine_id in zip(time_row, machine_row):
            job.append([(int(machine_id), int(processing_time))])
        instance.append(job)

    validate_instance(instance)
    return instance


def legacy_dict_to_fjssp(instance_dict: Dict[str, Any]) -> Instance:
    """Convert a legacy JSSP dict with ``times`` and ``machines`` to unified format."""
    if "times" not in instance_dict or "machines" not in instance_dict:
        raise KeyError("Legacy JSSP dict must contain 'times' and 'machines'")
    return tensor_jssp_to_fjssp(instance_dict["times"], instance_dict["machines"])


def fjssp_to_legacy_lists(instance: Instance) -> Dict[str, List[List[int]]]:
    """Convert single-option FJSSP/JSSP data to legacy machines/times lists.

    This helper is intentionally strict: every operation must have exactly one
    machine option, otherwise the instance is not a classic JSSP instance.
    """
    validate_instance(instance)
    times: List[List[int]] = []
    machines: List[List[int]] = []

    for job_id, job in enumerate(instance):
        time_row: List[int] = []
        machine_row: List[int] = []
        for op_idx, operation in enumerate(job):
            if len(operation) != 1:
                raise ValueError(
                    "Cannot convert flexible operation to classic JSSP at "
                    f"job {job_id}, op {op_idx}: {operation}"
                )
            machine_id, processing_time = operation[0]
            machine_row.append(machine_id)
            time_row.append(processing_time)
        machines.append(machine_row)
        times.append(time_row)

    return {"times": times, "machines": machines}


def fjssp_to_legacy_tensors(instance: Instance) -> Dict[str, Any]:
    """Convert single-option unified data to legacy torch tensors."""
    legacy_lists = fjssp_to_legacy_lists(instance)
    if torch is None:
        return legacy_lists
    return {
        "times": torch.tensor(legacy_lists["times"], dtype=torch.float32),
        "machines": torch.tensor(legacy_lists["machines"], dtype=torch.long),
    }
