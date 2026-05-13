"""Random JSSP generator that emits and saves unified FJSSP-compatible data."""

import os
import pickle
import random
from typing import Dict, List, Optional, Tuple

from .common import Instance, validate_instance

DEFAULT_JSSP_SIZES: List[Tuple[int, int]] = [
    (6, 6),
    (10, 10),
    (20, 15),
    (20, 20),
    (30, 15),
    (30, 20),
    (50, 15),
    (50, 20),
    (100, 20),
]

def generate_random_jssp(
    num_jobs: Optional[int] = None,
    num_machines: Optional[int] = None,
    sizes: List[Tuple[int, int]] = DEFAULT_JSSP_SIZES,
    instances_per_size: int = 1, #change for full dataset generation
    processing_time_range: Tuple[int, int] = (1, 99),
    seed: Optional[int] = None,
    save_dir: Optional[str] = None,
) -> Instance | List[Instance] | Dict[Tuple[int, int], List[Instance]]:
    """Generate unified JSSP data for one size or many sizes, then save it."""

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
        generated_instances: List[Instance] = []
        for _ in range(instances_per_size):
            instance: Instance = []
            for _ in range(jobs_count):
                machine_order = list(range(machines_count))
                rng.shuffle(machine_order)
                job = []
                for machine_id in machine_order:
                    processing_time = rng.randint(min_time, max_time)
                    job.append([(machine_id, processing_time)])
                instance.append(job)
            validate_instance(instance, num_machines=machines_count)
            generated_instances.append(instance)

        datasets[(jobs_count, machines_count)] = generated_instances
        filename = f"jssp_synthetic_new_{jobs_count}x{machines_count}_{instances_per_size}.pkl"
        save_payload = generated_instances[0] if instances_per_size == 1 else generated_instances
        with open(os.path.join(output_dir, filename), "wb") as handle:
            pickle.dump(save_payload, handle)
        full_path = os.path.join(output_dir, filename)
        print(f"Saved to {full_path}")    

    if single_size_requested:
        only_instances = datasets[(num_jobs, num_machines)]
        return only_instances[0] if instances_per_size == 1 else only_instances

    return datasets
