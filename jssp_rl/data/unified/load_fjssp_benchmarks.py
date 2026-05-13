"""One-time converter for SchedulingLab FJSSP benchmark text files."""

import json
import os
import pickle
import urllib.request
from typing import Dict, List

from .common import Instance, validate_instance

GITHUB_TREE_URL = "https://api.github.com/repos/SchedulingLab/fjsp-instances/git/trees/main?recursive=1"
RAW_BASE_URL = "https://raw.githubusercontent.com/SchedulingLab/fjsp-instances/main"
BENCHMARK_NAMES = (
    "barnes",
    "behnke",
    "brandimarte",
    "dauzere",
    "fattahi",
    "hurink",
    "kacem",
)


def parse_fjssp_text(text: str) -> Instance:
    """Parse one FJSSP text file into the unified nested-list format."""
    tokens = [int(token) for token in text.split()]
    cursor = 0
    num_jobs = tokens[cursor]
    cursor += 1
    num_machines = tokens[cursor]
    cursor += 1

    instance: Instance = []
    for _ in range(num_jobs):
        num_operations = tokens[cursor]
        cursor += 1
        job = []
        for _ in range(num_operations):
            num_options = tokens[cursor]
            cursor += 1
            operation = []
            for _ in range(num_options):
                machine_id = tokens[cursor]
                processing_time = tokens[cursor + 1]
                cursor += 2
                operation.append((machine_id, processing_time))
            operation.sort(key=lambda option: option[0])
            job.append(operation)
        instance.append(job)

    validate_instance(instance, num_machines=num_machines)
    return instance


def convert_fjssp_benchmarks(
    names=BENCHMARK_NAMES,
    save_dir: str | None = None,
) -> Dict[str, List[Instance]]:
    """Download FJSSP text files, convert them, and save benchmark pickle files."""
    output_dir = save_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "saved",
        "fjssp_benchmarks",
    )
    os.makedirs(output_dir, exist_ok=True)

    request = urllib.request.Request(GITHUB_TREE_URL, headers={"User-Agent": "jssp-rl-fjssp-loader"})
    with urllib.request.urlopen(request) as response:
        tree = json.loads(response.read().decode("utf-8"))

    text_paths = sorted(
        item["path"]
        for item in tree["tree"]
        if item.get("type") == "blob" and item.get("path", "").endswith(".txt")
    )

    converted: Dict[str, List[Instance]] = {}
    for name in names:
        benchmark_name = name.lower()
        instances: List[Instance] = []
        for path in text_paths:
            if not path.lower().startswith(f"{benchmark_name}/"):
                continue
            url = f"{RAW_BASE_URL}/{path}"
            request = urllib.request.Request(url, headers={"User-Agent": "jssp-rl-fjssp-loader"})
            with urllib.request.urlopen(request) as response:
                instances.append(parse_fjssp_text(response.read().decode("utf-8")))

        if not instances:
            continue

        with open(os.path.join(output_dir, f"benchmark_{benchmark_name}.pkl"), "wb") as handle:
            pickle.dump(instances, handle)
        converted[benchmark_name] = instances

    return converted


if __name__ == "__main__":
    convert_fjssp_benchmarks()
