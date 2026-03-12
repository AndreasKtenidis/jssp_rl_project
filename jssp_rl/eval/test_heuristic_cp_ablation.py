
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from typing import Iterable, List, Optional

import pandas as pd
import torch

# Make repo root importable when script is launched as a file.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from cp.cp_solver import solve_instance_your_version, solve_with_rl_warmstart
from utils.dispatching_heuristics import generate_pdr_schedule, supported_rules


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _gap(ms, bks):
    if ms is None or bks is None or bks <= 0:
        return None
    return round((float(ms) - float(bks)) / float(bks) * 100.0, 2)


def _parse_fix_ratios(raw: Iterable[float]) -> List[float]:
    out = []
    for r in raw:
        v = float(r)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Invalid fix ratio {r}; expected in [0, 1].")
        out.append(v)
    return out


def run_heuristic_cp_ablation(
    rules: List[str],
    fix_ratios: List[float],
    num_instances: Optional[int],
    start_index: int,
    time_limit: float,
    run_pure_cp: bool = True,
):
    base_dir = BASE_DIR
    saved_dir = os.path.join(base_dir, "saved")
    eval_dir = os.path.join(base_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    data_path = os.path.join(saved_dir, "benchmark_taillard.pkl")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found. Run: python data/download_benchmarks.py --fetch"
        )

    with open(data_path, "rb") as f:
        instances = pickle.load(f)

    if start_index < 0 or start_index >= len(instances):
        raise ValueError(f"start_index={start_index} is out of range for {len(instances)} instances")

    if num_instances is None:
        selected = instances[start_index:]
    else:
        selected = instances[start_index:start_index + num_instances]

    print("=" * 72)
    print("Heuristic+CP Taillard Ablation")
    print("=" * 72)
    print(f"Instances selected: {len(selected)} (from index {start_index})")
    print(f"Rules: {rules}")
    print(f"Fix ratios: {fix_ratios}")
    print(f"CP time limit: {time_limit}s")
    print(f"Pure CP baseline: {'ON' if run_pure_cp else 'OFF'}")

    rows = []

    for idx, inst in enumerate(selected, 1):
        name = inst.get("name", f"ta_{idx}")
        times_np = _to_numpy(inst["times"])
        machines_np = _to_numpy(inst["machines"])
        n_jobs, n_machines = times_np.shape
        bks = inst.get("bks", -1)

        print(f"\n[{idx:02d}/{len(selected):02d}] {name} ({n_jobs}x{n_machines}) | BKS={bks}")

        cp_ms = None
        cp_time = None
        if run_pure_cp:
            t0 = time.time()
            cp_ms, _ = solve_instance_your_version(times_np, machines_np, time_limit_s=time_limit)
            cp_time = time.time() - t0
            print(f"  CP cold: MS={cp_ms} ({cp_time:.2f}s)")

        for rule in rules:
            t0 = time.time()
            heur_ms, heur_sched = generate_pdr_schedule(times_np, machines_np, rule=rule)
            heur_time = time.time() - t0
            print(f"  Heuristic {rule.upper()}: MS={heur_ms} ({heur_time:.4f}s)")

            for fr in fix_ratios:
                t1 = time.time()
                warm_ms, _ = solve_with_rl_warmstart(
                    times_np,
                    machines_np,
                    rl_schedule=heur_sched,
                    time_limit_s=time_limit,
                    fix_ratio=fr,
                )
                warm_time = time.time() - t1

                rows.append(
                    {
                        "instance": name,
                        "size": f"{n_jobs}x{n_machines}",
                        "bks": bks,
                        "rule": rule,
                        "fix_ratio": fr,
                        "heuristic_ms": heur_ms,
                        "heuristic_time_s": round(heur_time, 4),
                        "cp_cold_ms": cp_ms,
                        "cp_cold_time_s": round(cp_time, 3) if cp_time is not None else None,
                        "heuristic_cp_ms": warm_ms,
                        "heuristic_cp_time_s": round(warm_time, 3),
                        "gap_heuristic_pct": _gap(heur_ms, bks),
                        "gap_cp_cold_pct": _gap(cp_ms, bks),
                        "gap_heuristic_cp_pct": _gap(warm_ms, bks),
                    }
                )
                print(f"    fix={int(fr * 100):>2d}% -> MS={warm_ms} ({warm_time:.2f}s)")

    df = pd.DataFrame(rows)
    raw_out = os.path.join(eval_dir, "heuristic_cp_taillard_fix_ratio_results.csv")
    df.to_csv(raw_out, index=False)

    summary = (
        df.groupby(["rule", "fix_ratio"], as_index=False)
        .agg(
            heuristic_ms_mean=("heuristic_ms", "mean"),
            cp_cold_ms_mean=("cp_cold_ms", "mean"),
            heuristic_cp_ms_mean=("heuristic_cp_ms", "mean"),
            heuristic_time_s_mean=("heuristic_time_s", "mean"),
            cp_cold_time_s_mean=("cp_cold_time_s", "mean"),
            heuristic_cp_time_s_mean=("heuristic_cp_time_s", "mean"),
            gap_heuristic_pct_mean=("gap_heuristic_pct", "mean"),
            gap_cp_cold_pct_mean=("gap_cp_cold_pct", "mean"),
            gap_heuristic_cp_pct_mean=("gap_heuristic_cp_pct", "mean"),
            num_instances=("instance", "nunique"),
        )
        .round(3)
    )
    summary_out = os.path.join(eval_dir, "heuristic_cp_taillard_fix_ratio_summary.csv")
    summary.to_csv(summary_out, index=False)

    print("\n" + "=" * 72)
    print("Summary (mean over selected instances)")
    print("=" * 72)
    print(summary.to_string(index=False))
    print(f"\nSaved raw results: {raw_out}")
    print(f"Saved summary:     {summary_out}")
    return df, summary


def main():
    parser = argparse.ArgumentParser(description="Run heuristic+CP fix-ratio ablation on Taillard.")
    parser.add_argument(
        "--rules",
        nargs="+",
        default=["mwkr", "spt", "fifo"],
        help="Dispatching rules to evaluate. Supported: mwkr spt fifo",
    )
    parser.add_argument(
        "--fix-ratios",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4],
        help="Fractions of schedule to hard-fix in CP.",
    )
    parser.add_argument("--num-instances", type=int, default=None, help="How many Taillard instances to run.")
    parser.add_argument("--start-index", type=int, default=0, help="Index offset in benchmark_taillard.pkl.")
    parser.add_argument("--time-limit", type=float, default=30.0, help="CP time limit in seconds.")
    parser.add_argument(
        "--skip-pure-cp",
        action="store_true",
        help="Skip cold CP baseline and only run heuristic+CP.",
    )
    args = parser.parse_args()

    valid = set(supported_rules())
    rules = []
    for rule in args.rules:
        rule_l = rule.lower().strip()
        if rule_l not in valid and rule_l not in {"mtwr", "most_work_remaining", "most_total_work_remaining"}:
            raise ValueError(f"Unsupported rule '{rule}'. Supported: {sorted(valid)} (+ mtwr alias)")
        rules.append(rule_l)

    fix_ratios = _parse_fix_ratios(args.fix_ratios)
    run_heuristic_cp_ablation(
        rules=rules,
        fix_ratios=fix_ratios,
        num_instances=args.num_instances,
        start_index=args.start_index,
        time_limit=args.time_limit,
        run_pure_cp=not args.skip_pure_cp,
    )


if __name__ == "__main__":
    main()
