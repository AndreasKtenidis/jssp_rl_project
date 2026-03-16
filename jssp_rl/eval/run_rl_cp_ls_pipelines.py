"""
run_rl_cp_ls_pipelines.py
=========================
Evaluate RL-CP-LS pipelines by:
1) Selecting top RL-CP combinations from an existing results CSV.
2) Re-running RL inference + chosen RL-CP baseline on benchmark PKLs.
3) Applying CP-based local-search refinement pipelines (LNS / NS-inspired).
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import time
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.distributions import Categorical

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.Hetero_actor_critic_ppo import ActorCriticPPO
from models.gin import HeteroGATv2
from env.jssp_environment import JSSPEnvironment
from utils.features import make_hetero_data
from cp.cp_solver import (
    solve_hybrid_advanced,
    solve_with_rl_warmstart,
    run_lns_refinement,
    validate_schedule,
)


BENCHMARKS = {
    "FT": "benchmark_ft.pkl",
    "Taillard": "benchmark_taillard.pkl",
    "DMU": "benchmark_dmu.pkl",
    "ABZ": "benchmark_abz.pkl",
    "Lawrence": "benchmark_la.pkl",
    "ORB": "benchmark_orb.pkl",
    "SWV": "benchmark_swv.pkl",
    "YN": "benchmark_yn.pkl",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cp_time_limit(n_jobs: int, n_machines: int) -> float:
    ops = n_jobs * n_machines
    if ops <= 225:
        return 45.0
    if ops <= 600:
        return 90.0
    return 180.0


def gap(ms, bks):
    if ms is None or bks is None or bks <= 0:
        return None
    return round((float(ms) - float(bks)) / float(bks) * 100.0, 3)


def _single_rollout(env, model, static_edges, stochastic: bool):
    env.reset()
    done = False
    while not done:
        data = make_hetero_data(env, DEVICE, precomputed_edge_index=static_edges)
        logits, _ = model(data)
        avail = env.get_available_actions()
        if not avail:
            return float("inf"), []
        mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=DEVICE)
        mask[avail] = True
        logits = logits.masked_fill(~mask, -1e10)
        if stochastic:
            action = Categorical(logits=logits).sample().item()
        else:
            action = logits.argmax().item()
        _, _, done, _ = env.step(action)
    return env.get_makespan(), env.extract_job_assignments()


@torch.no_grad()
def run_rl_best_of_k(instance, model, best_of_k: int = 10):
    times = instance["times"]
    machines = instance["machines"]
    if not isinstance(times, torch.Tensor):
        times = torch.tensor(times)
    if not isinstance(machines, torch.Tensor):
        machines = torch.tensor(machines)
    times = times.to(DEVICE)
    machines = machines.to(DEVICE)

    env = JSSPEnvironment(times, machines, device=DEVICE, use_shaping_rewards=False)
    static_edges = HeteroGATv2.build_edge_index_dict(env.machines)

    t0 = time.time()
    greedy_ms, greedy_sched = _single_rollout(env, model, static_edges, stochastic=False)
    greedy_t = time.time() - t0

    bok_ms, bok_sched = greedy_ms, greedy_sched
    t1 = time.time()
    for _ in range(max(1, int(best_of_k))):
        ms, sched = _single_rollout(env, model, static_edges, stochastic=True)
        if ms < bok_ms:
            bok_ms, bok_sched = ms, sched
    bok_t = time.time() - t1

    return greedy_ms, greedy_sched, greedy_t, bok_ms, bok_sched, bok_t


def parse_combo_col(col_name: str):
    """Map a gap column to solver mode."""
    m = re.fullmatch(r"Gap_Hybrid_(\d+)pct", col_name)
    if m:
        pct = int(m.group(1))
        return {
            "gap_col": col_name,
            "combo_id": f"Hybrid_{pct}pct",
            "kind": "warmstart",
            "fix_ratio": pct / 100.0,
            "mode": None,
        }

    m = re.fullmatch(r"Gap_Hybrid_CPFix_(\d+)pct", col_name)
    if m:
        pct = int(m.group(1))
        return {
            "gap_col": col_name,
            "combo_id": f"Hybrid_CPFix_{pct}pct",
            "kind": "advanced",
            "fix_ratio": pct / 100.0,
            "mode": "cp_fixation",
        }

    if col_name == "Gap_Hybrid_CPHint":
        return {
            "gap_col": col_name,
            "combo_id": "Hybrid_CPHint",
            "kind": "advanced",
            "fix_ratio": 0.0,
            "mode": "cp_hint",
        }
    if col_name == "Gap_Hybrid_MachPrec":
        return {
            "gap_col": col_name,
            "combo_id": "Hybrid_MachPrec",
            "kind": "advanced",
            "fix_ratio": 0.0,
            "mode": "machine_precedence",
        }

    return None


def pick_top_combos(results_csv: str, top_k: int = 3, combo_cols: Optional[List[str]] = None):
    df = pd.read_csv(results_csv)
    if combo_cols:
        gap_cols = combo_cols
    else:
        gap_cols = [c for c in df.columns if c.startswith("Gap_Hybrid_")]
    if not gap_cols:
        raise ValueError("No Gap_Hybrid_* columns found.")

    means = {}
    for c in gap_cols:
        means[c] = pd.to_numeric(df[c], errors="coerce").mean()
    ranked = sorted(means.items(), key=lambda x: x[1])

    selected = []
    for col, mean_gap in ranked:
        parsed = parse_combo_col(col)
        if parsed is None:
            continue
        parsed["mean_gap_from_csv"] = float(mean_gap)
        selected.append(parsed)
        if len(selected) >= int(top_k):
            break
    if not selected:
        raise ValueError("No supported hybrid combinations were selected.")
    return selected, ranked


def run_base_hybrid(instance, combo_cfg, rl_schedule, time_limit_s: float, bks):
    times = instance["times"]
    machines = instance["machines"]
    t_np = times.numpy() if hasattr(times, "numpy") else times
    m_np = machines.numpy() if hasattr(machines, "numpy") else machines

    t0 = time.time()
    if combo_cfg["kind"] == "warmstart":
        ms, sched, status, t_best = solve_with_rl_warmstart(
            t_np,
            m_np,
            rl_schedule=rl_schedule,
            time_limit_s=time_limit_s,
            fix_ratio=float(combo_cfg["fix_ratio"]),
            bks=bks,
        )
    else:
        ms, sched, status, t_best = solve_hybrid_advanced(
            t_np,
            m_np,
            rl_schedule=rl_schedule,
            mode=combo_cfg["mode"],
            fix_ratio=float(combo_cfg["fix_ratio"]),
            time_limit_s=time_limit_s,
            bks=bks,
        )
    wall_t = time.time() - t0
    return ms, sched, status, t_best, wall_t


def load_model(model_dir: str):
    ckpt_path = None
    for cand in [
        "best_ppo_latest.pt",
        "best_ppo_Phase_8.pt",
        "best_ppo_Phase_7.pt",
        "best_ppo_Phase_6.pt",
        "best_ppo_Phase_5.pt",
        "best_ppo.pt",
    ]:
        p = os.path.join(model_dir, cand)
        if os.path.exists(p):
            ckpt_path = p
            break
    if not ckpt_path:
        raise FileNotFoundError("No trained PPO checkpoint found in outputs/models.")

    model = ActorCriticPPO(3, 64, 32, 32, 32).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
    model.eval()
    return model, ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Run RL-CP-LS evaluation on benchmark PKLs.")
    parser.add_argument("--results-csv", type=str, default=os.path.join(BASE_DIR, "full_benchmark_adv_all.csv"))
    parser.add_argument("--top-k", type=int, default=3, help="Number of best RL-CP combos to keep from CSV.")
    parser.add_argument(
        "--combo-cols",
        nargs="+",
        default=None,
        help="Optional explicit Gap_Hybrid_* columns to use instead of auto top-k.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=["ns_lns_mix", "lns_machine", "lns_job", "lns_random"],
        help="Local-search pipelines from cp.cp_solver.available_lns_pipelines().",
    )
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS.keys()))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-instances-per-benchmark", type=int, default=None)
    parser.add_argument("--best-of-k", type=int, default=10)
    parser.add_argument("--ls-iters", type=int, default=8)
    parser.add_argument("--ls-free-ratio", type=float, default=0.10)
    parser.add_argument("--ls-step-time", type=float, default=6.0)
    parser.add_argument("--time-limit-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=os.path.join(BASE_DIR, "eval"))
    parser.add_argument("--out-prefix", type=str, default="rl_cp_ls")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"{args.out_prefix}_results.csv")
    out_summary = os.path.join(args.out_dir, f"{args.out_prefix}_summary.csv")

    combos, ranked = pick_top_combos(args.results_csv, top_k=args.top_k, combo_cols=args.combo_cols)
    print("\nTop RL-CP combinations from CSV:")
    for c in combos:
        print(
            f"  {c['combo_id']:<18s} | source={c['gap_col']:<24s} "
            f"| mean_gap={c['mean_gap_from_csv']:.3f}%"
        )

    print("\nFull ranked gap columns (best first):")
    for col, val in ranked[:12]:
        print(f"  {col:<28s} {val:.3f}%")

    model_dir = os.path.join(BASE_DIR, "outputs", "models")
    model, ckpt_path = load_model(model_dir)
    print(f"\nUsing checkpoint: {os.path.basename(ckpt_path)}")

    saved_dir = os.path.join(BASE_DIR, "saved")
    rows = []

    for bm in args.benchmarks:
        if bm not in BENCHMARKS:
            print(f"[WARN] Unknown benchmark '{bm}', skipping.")
            continue
        data_path = os.path.join(saved_dir, BENCHMARKS[bm])
        if not os.path.exists(data_path):
            print(f"[WARN] Missing {data_path}, skipping {bm}.")
            continue

        with open(data_path, "rb") as f:
            instances = pickle.load(f)

        subset = instances[args.start_index:]
        if args.max_instances_per_benchmark is not None:
            subset = subset[: args.max_instances_per_benchmark]

        print(f"\n=== Benchmark {bm}: {len(subset)} instances ===")
        for i, inst in enumerate(subset, 1):
            name = inst.get("name", f"{bm}_{i}")
            times = inst["times"]
            machines = inst["machines"]
            n_jobs, n_machines = times.shape
            bks = inst.get("bks", -1)
            base_tl = cp_time_limit(n_jobs, n_machines) * float(args.time_limit_scale)

            print(f"\n[{bm} {i:03d}/{len(subset):03d}] {name} ({n_jobs}x{n_machines}) BKS={bks} TL={base_tl:.1f}s")

            # RL inference once per instance
            rl_start = time.time()
            greedy_ms, _, greedy_t, bok_ms, bok_sched, bok_t = run_rl_best_of_k(
                inst, model, best_of_k=args.best_of_k
            )
            rl_wall = time.time() - rl_start
            print(f"  RL: greedy={greedy_ms:.0f}, BoK={bok_ms:.0f}, rollout_total={rl_wall:.2f}s")

            t_np = times.numpy() if hasattr(times, "numpy") else times
            m_np = machines.numpy() if hasattr(machines, "numpy") else machines

            for combo in combos:
                base_ms, base_sched, base_status, base_t_best, base_wall = run_base_hybrid(
                    inst, combo, bok_sched, time_limit_s=base_tl, bks=bks
                )
                base_valid, base_msg = validate_schedule(base_sched, t_np, m_np) if base_sched else (False, "no schedule")
                print(
                    f"  Base {combo['combo_id']}: MS={base_ms} status={base_status} "
                    f"valid={base_valid} ({base_wall:.2f}s)"
                )

                for pipeline in args.pipelines:
                    ls_start = time.time()
                    if base_sched and base_valid and base_ms is not None:
                        ls_ms, ls_sched, hist = run_lns_refinement(
                            t_np,
                            m_np,
                            initial_schedule=base_sched,
                            pipeline=pipeline,
                            iterations=args.ls_iters,
                            free_ratio=args.ls_free_ratio,
                            step_time_s=args.ls_step_time,
                            seed=args.seed + i,
                            bks=bks,
                        )
                        ls_valid, ls_msg = validate_schedule(ls_sched, t_np, m_np) if ls_sched else (False, "no schedule")
                    else:
                        ls_ms, ls_sched, hist = None, [], []
                        ls_valid, ls_msg = False, "invalid base schedule"

                    ls_wall = time.time() - ls_start

                    improved = (
                        base_ms is not None
                        and ls_ms is not None
                        and float(ls_ms) < float(base_ms)
                    )
                    impr_abs = (float(base_ms) - float(ls_ms)) if improved else 0.0
                    impr_pct = (impr_abs / float(base_ms) * 100.0) if improved and float(base_ms) > 0 else 0.0
                    accepted_iters = sum(1 for h in hist if h.get("accepted"))

                    print(
                        f"    LS {pipeline:<12s}: MS={ls_ms} valid={ls_valid} "
                        f"improved={improved} (+{impr_abs:.1f}) [{ls_wall:.2f}s]"
                    )

                    rows.append(
                        {
                            "Benchmark": bm,
                            "Instance": name,
                            "Size": f"{n_jobs}x{n_machines}",
                            "BKS": bks,
                            "Combo": combo["combo_id"],
                            "Combo_Source_Col": combo["gap_col"],
                            "Combo_Mean_Gap_From_CSV": round(combo["mean_gap_from_csv"], 4),
                            "Pipeline": pipeline,
                            "RL_Greedy_MS": greedy_ms,
                            "RL_BoK_MS": bok_ms,
                            "RL_Greedy_Time_s": round(greedy_t, 4),
                            "RL_BoK_Time_s": round(bok_t, 4),
                            "Base_MS": base_ms,
                            "Base_Status": base_status,
                            "Base_TimeToBest_s": round(float(base_t_best), 4),
                            "Base_WallTime_s": round(base_wall, 4),
                            "Base_Valid": base_valid,
                            "Base_Valid_Msg": base_msg,
                            "LS_MS": ls_ms,
                            "LS_WallTime_s": round(ls_wall, 4),
                            "LS_Valid": ls_valid,
                            "LS_Valid_Msg": ls_msg,
                            "Gap_Base_pct": gap(base_ms, bks),
                            "Gap_LS_pct": gap(ls_ms, bks),
                            "Improved": improved,
                            "Improve_Abs": round(impr_abs, 4),
                            "Improve_PctOfBase": round(impr_pct, 4),
                            "LS_Accepted_Iters": accepted_iters,
                            "LS_Total_Iters": len(hist),
                            "CP_Time_Limit_s": round(base_tl, 3),
                            "LS_Step_Time_s": args.ls_step_time,
                            "LS_Free_Ratio": args.ls_free_ratio,
                        }
                    )

            # incremental save
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo rows produced.")
        return

    summary = (
        df.groupby(["Combo", "Pipeline"], as_index=False)
        .agg(
            n_instances=("Instance", "count"),
            base_ms_mean=("Base_MS", "mean"),
            ls_ms_mean=("LS_MS", "mean"),
            gap_base_mean=("Gap_Base_pct", "mean"),
            gap_ls_mean=("Gap_LS_pct", "mean"),
            improved_rate=("Improved", "mean"),
            improve_abs_mean=("Improve_Abs", "mean"),
            base_time_mean=("Base_WallTime_s", "mean"),
            ls_time_mean=("LS_WallTime_s", "mean"),
            valid_ls_rate=("LS_Valid", "mean"),
        )
        .round(4)
    )

    summary.to_csv(out_summary, index=False)
    df.to_csv(out_csv, index=False)

    print(f"\nSaved detailed results: {out_csv}")
    print(f"Saved summary:         {out_summary}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
