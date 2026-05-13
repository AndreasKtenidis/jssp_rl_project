"""Random FJSSP generator that emits and saves unified-format data."""

import os
import pickle
import random
from typing import Dict, List, Optional, Tuple

from .common import Instance, validate_instance

DEFAULT_FJSSP_FLEXIBILITY_RANGES: Dict[Tuple[int, int], Tuple[float, float]] = {
    (6, 6): (0.15, 0.30),
    (10, 5): (0.20, 0.40),
    (10, 10): (0.20, 0.50),
    (15, 10): (0.30, 0.50),
    (20, 10): (0.30, 0.60),
    (20, 15): (0.40, 0.70),
}

DEFAULT_FJSSP_SIZES: List[Tuple[int, int]] = list(DEFAULT_FJSSP_FLEXIBILITY_RANGES.keys())


def generate_random_fjssp(
    num_jobs: Optional[int] = None,
    num_machines: Optional[int] = None,
    sizes: List[Tuple[int, int]] = DEFAULT_FJSSP_SIZES,
    flexibility_ranges: Dict[Tuple[int, int], Tuple[float, float]] = DEFAULT_FJSSP_FLEXIBILITY_RANGES,
    instances_per_size: int = 1,
    processing_time_range: Tuple[int, int] = (1, 99),
    seed: Optional[int] = None,
    save_dir: Optional[str] = None,
) -> Instance | List[Instance] | Dict[Tuple[int, int], List[Instance]]:
    """Generate unified FJSSP data for one size or many sizes, then save it."""
    min_time, max_time = processing_time_range
    rng = random.Random(seed)
    single_size_requested = num_jobs is not None and num_machines is not None
    selected_sizes = [(num_jobs, num_machines)] if single_size_requested else sizes
    output_dir = save_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "saved",
    )
    os.makedirs(output_dir, exist_ok=True)

    datasets: Dict[Tuple[int, int], List[Instance]] = {}
    for jobs_count, machines_count in selected_sizes:
        min_flex, max_flex = flexibility_ranges.get((jobs_count, machines_count), (0.30, 0.60))
        generated_instances: List[Instance] = []

        for _ in range(instances_per_size):
            instance: Instance = []
            for _ in range(jobs_count):
                job = []
                for _ in range(machines_count):
                    flexibility = rng.uniform(min_flex, max_flex)
                    alternatives_count = max(1, round(flexibility * machines_count))
                    alternatives_count = min(alternatives_count, machines_count)
                    eligible_machines = rng.sample(range(machines_count), alternatives_count)
                    operation = [
                        (machine_id, rng.randint(min_time, max_time))
                        for machine_id in eligible_machines
                    ]
                    operation.sort(key=lambda option: option[0])
                    job.append(operation)
                instance.append(job)

            validate_instance(instance, num_machines=machines_count)
            generated_instances.append(instance)

        datasets[(jobs_count, machines_count)] = generated_instances
        filename = f"fjssp_synthetic_new_{jobs_count}x{machines_count}_{instances_per_size}.pkl"
        save_payload = generated_instances[0] if instances_per_size == 1 else generated_instances
        with open(os.path.join(output_dir, filename), "wb") as handle:
            pickle.dump(save_payload, handle)

    if single_size_requested:
        only_instances = datasets[(num_jobs, num_machines)]
        return only_instances[0] if instances_per_size == 1 else only_instances

    return datasets
