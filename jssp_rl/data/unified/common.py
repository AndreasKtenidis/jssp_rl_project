"""Common data types and validation helpers for scheduling instances.

The unified representation is FJSSP-first:

    instance[job_id][operation_index] -> [(machine_id, processing_time), ...]

Classic JSSP is represented as the special case where each operation has
exactly one machine option.
"""

from typing import List, Optional, Tuple

MachineOption = Tuple[int, int]
Operation = List[MachineOption]
Job = List[Operation]
Instance = List[Job]


def infer_num_jobs(instance: Instance) -> int:
    """Return the number of jobs in a unified scheduling instance."""
    return len(instance)


def infer_num_machines(instance: Instance) -> int:
    """Infer the number of machines from the maximum machine id used."""
    max_machine_id: Optional[int] = None
    for job in instance:
        for operation in job:
            for machine_id, _ in operation:
                if max_machine_id is None or machine_id > max_machine_id:
                    max_machine_id = machine_id
    return 0 if max_machine_id is None else max_machine_id + 1


def validate_machine_ids(instance: Instance, num_machines: Optional[int] = None) -> None:
    """Validate that all machine ids are non-negative and within bounds."""
    for job_id, job in enumerate(instance):
        for op_idx, operation in enumerate(job):
            for machine_id, _ in operation:
                if not isinstance(machine_id, int):
                    raise TypeError(
                        f"Machine id must be int at job {job_id}, op {op_idx}: {machine_id!r}"
                    )
                if machine_id < 0:
                    raise ValueError(
                        f"Machine id must be non-negative at job {job_id}, op {op_idx}: {machine_id}"
                    )
                if num_machines is not None and machine_id >= num_machines:
                    raise ValueError(
                        f"Machine id {machine_id} at job {job_id}, op {op_idx} "
                        f"is outside num_machines={num_machines}"
                    )


def validate_processing_times(instance: Instance) -> None:
    """Validate that all processing times are positive integers."""
    for job_id, job in enumerate(instance):
        for op_idx, operation in enumerate(job):
            for _, processing_time in operation:
                if not isinstance(processing_time, int):
                    raise TypeError(
                        "Processing time must be int at "
                        f"job {job_id}, op {op_idx}: {processing_time!r}"
                    )
                if processing_time <= 0:
                    raise ValueError(
                        "Processing time must be positive at "
                        f"job {job_id}, op {op_idx}: {processing_time}"
                    )


def validate_instance(instance: Instance, num_machines: Optional[int] = None) -> None:
    """Validate the basic structure and values of a unified instance."""
    if not isinstance(instance, list):
        raise TypeError("Instance must be a list of jobs")
    if len(instance) == 0:
        raise ValueError("Instance must contain at least one job")

    for job_id, job in enumerate(instance):
        if not isinstance(job, list):
            raise TypeError(f"Job {job_id} must be a list of operations")
        if len(job) == 0:
            raise ValueError(f"Job {job_id} must contain at least one operation")
        for op_idx, operation in enumerate(job):
            if not isinstance(operation, list):
                raise TypeError(f"Operation {job_id}:{op_idx} must be a list of machine options")
            if len(operation) == 0:
                raise ValueError(f"Operation {job_id}:{op_idx} must have at least one machine option")
            seen_machines = set()
            for option in operation:
                if not isinstance(option, tuple) or len(option) != 2:
                    raise TypeError(
                        f"Machine option at job {job_id}, op {op_idx} must be a (machine, time) tuple"
                    )
                machine_id, _ = option
                if machine_id in seen_machines:
                    raise ValueError(
                        f"Duplicate machine option {machine_id} at job {job_id}, op {op_idx}"
                    )
                seen_machines.add(machine_id)

    validate_machine_ids(instance, num_machines=num_machines)
    validate_processing_times(instance)
