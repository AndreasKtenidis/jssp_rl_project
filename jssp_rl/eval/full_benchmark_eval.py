"""
full_benchmark_eval.py
======================
Press Play — this script generates complete paper-ready tables for:
- RL Greedy
- Best-of-K (K=10) RL
- Pure CP (cold start, 30s)
- Hybrid (RL best-of-K warm-start + CP) for fix ratios: 0%, 10%, 20%, 30%, 40%

Columns per instance:
  Name | Size | BKS | RL_Greedy | RL_BoK | CP | Hybrid_0% | Hybrid_10% | ...
  RL_Time | CP_Time | Hybrid_Time | Gap_RL | Gap_BoK | Gap_CP | Gap_Hybrid_X%

Outputs:
  eval/full_benchmark_<BenchmarkName>.csv  — one per benchmark
  eval/full_benchmark_all.csv              — combined
"""

import os, sys, pickle, json, time
import torch
import pandas as pd
from torch.distributions import Categorical

# ─── Path setup ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from models.Hetero_actor_critic_ppo import ActorCriticPPO
from models.gin import HeteroGATv2
from env.jssp_environment import JSSPEnvironment
from utils.features import make_hetero_data
from cp.cp_solver import solve_with_rl_warmstart, solve_instance_your_version

# ─── Configuration ──────────────────────────────────────────────────────────
BEST_OF_K    = 10
FIX_RATIOS   = [0.0, 0.1, 0.2, 0.3, 0.4]   # 0%, 10%, 20%, 30%, 40%
CP_TIME_BASE = 30    # seconds for small–medium instances

# Adaptive CP time limit per instance size
def cp_time_limit(n, m):
    ops = n * m
    if   ops <= 36:   return 10
    elif ops <= 150:  return 20
    elif ops <= 600:  return 30
    elif ops <= 1200: return 60
    else:             return 120

# Benchmark files (name → filename)
BENCHMARKS = {
    "FT":       "benchmark_ft.pkl",
    "Taillard": "benchmark_taillard.pkl",
    "DMU":      "benchmark_dmu.pkl",
    "ABZ":      "benchmark_abz.pkl",
    "Lawrence": "benchmark_la.pkl",
    "ORB":      "benchmark_orb.pkl",
    "SWV":      "benchmark_swv.pkl",
    "YN":       "benchmark_yn.pkl",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── RL Rollout helpers ──────────────────────────────────────────────────────

@torch.no_grad()
def _single_rollout(env, model, static_edges, stochastic: bool):
    """One greedy or stochastic rollout. Returns (makespan, schedule)."""
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
def run_rl(instance, model):
    """
    Returns:
      greedy_ms, greedy_sched    — single deterministic rollout
      bok_ms,    bok_sched       — best of BEST_OF_K stochastic rollouts
      rl_time_single             — wall time for ONE greedy rollout (seconds)
    """
    times    = instance["times"]
    machines = instance["machines"]
    if not isinstance(times,    torch.Tensor): times    = torch.tensor(times)
    if not isinstance(machines, torch.Tensor): machines = torch.tensor(machines)
    times    = times.to(DEVICE)
    machines = machines.to(DEVICE)

    env          = JSSPEnvironment(times, machines, device=DEVICE, use_shaping_rewards=False)
    static_edges = HeteroGATv2.build_edge_index_dict(env.machines)

    # Greedy (timed = rl_time_single)
    t0 = time.time()
    greedy_ms, greedy_sched = _single_rollout(env, model, static_edges, stochastic=False)
    rl_time_single = time.time() - t0

    # Best-of-K stochastic
    bok_ms, bok_sched = greedy_ms, greedy_sched
    for _ in range(BEST_OF_K):
        ms, sched = _single_rollout(env, model, static_edges, stochastic=True)
        if ms < bok_ms:
            bok_ms, bok_sched = ms, sched

    return greedy_ms, greedy_sched, bok_ms, bok_sched, rl_time_single


def run_cp(instance, tl):
    """Pure CP cold start. Returns (makespan, solve_time)."""
    times    = instance["times"]
    machines = instance["machines"]
    t_np  = times.numpy()    if hasattr(times,    "numpy") else times
    m_np  = machines.numpy() if hasattr(machines, "numpy") else machines
    t0 = time.time()
    ms, _ = solve_instance_your_version(t_np, m_np, time_limit_s=tl)
    return ms, time.time() - t0


def run_hybrid(instance, schedule, tl, fix_ratio):
    """Hybrid CP with RL warm-start. Returns (makespan, solve_time)."""
    times    = instance["times"]
    machines = instance["machines"]
    t_np  = times.numpy()    if hasattr(times,    "numpy") else times
    m_np  = machines.numpy() if hasattr(machines, "numpy") else machines
    t0 = time.time()
    ms, _ = solve_with_rl_warmstart(t_np, m_np, rl_schedule=schedule,
                                    time_limit_s=tl, fix_ratio=fix_ratio)
    return ms, time.time() - t0


def gap(ms, bks):
    if bks > 0 and ms > 0:
        return round((ms - bks) / bks * 100, 1)
    return None


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    saved_dir = os.path.join(BASE_DIR, "saved")
    eval_dir  = os.path.join(BASE_DIR, "eval")
    model_dir = os.path.join(BASE_DIR, "outputs", "models")
    os.makedirs(eval_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_path = None
    for candidate in ["best_ppo_latest.pt", "best_ppo_Phase_7.pt", "best_ppo_Phase_6.pt",
                       "best_ppo_Phase_5.pt", "best_ppo_Phase_4.pt"]:
        p = os.path.join(model_dir, candidate)
        if os.path.exists(p):
            ckpt_path = p
            break

    if not ckpt_path:
        print("[ERROR] No model checkpoint found. Run training first.")
        return

    print(f"\n✅  Loading model: {os.path.basename(ckpt_path)}")
    model = ActorCriticPPO(3, 64, 32, 32, 32).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    all_rows = []

    for bm_name, bm_file in BENCHMARKS.items():
        path = os.path.join(saved_dir, bm_file)
        if not os.path.exists(path):
            print(f"\n[SKIP] {bm_name}: {bm_file} not found")
            continue

        with open(path, "rb") as f:
            instances = pickle.load(f)

        print(f"\n{'='*70}")
        print(f"  Benchmark: {bm_name}  ({len(instances)} instances)")
        print(f"{'='*70}")

        # ── Resume Logic ──────────────
        out_bm = os.path.join(eval_dir, f"full_benchmark_{bm_name.lower()}.csv")
        existing_instances = set()
        if os.path.exists(out_bm):
            try:
                df_existing = pd.read_csv(out_bm)
                existing_instances = set(df_existing["Instance"].tolist())
                bm_rows = df_existing.to_dict("records")
                print(f"  [Resume] Found existing results. Skipping {len(existing_instances)} instances.")
            except Exception:
                bm_rows = []
        else:
            bm_rows = []

        for i, inst in enumerate(instances):
            name = inst.get("name", f"{bm_name}_{i}")
            if name in existing_instances:
                continue

            times    = inst["times"]
            machines = inst["machines"]
            N, M = (times.shape if hasattr(times, "shape")
                    else (len(times), len(times[0])))
            bks = inst.get("bks", -1)
            tl  = cp_time_limit(N, M)

            print(f"\n  [{i+1:3d}] {name} ({N}x{M})  BKS={bks}  CP_TL={tl}s")

            # ── RL ────────────────────────────────────────────────────────────
            print("        RL ...", end=" ", flush=True)
            greedy_ms, greedy_sched, bok_ms, bok_sched, rl_t = run_rl(inst, model)
            print(f"Greedy={greedy_ms:.0f}  BoK={bok_ms:.0f}  ({rl_t:.2f}s/rollout)")

            # ── Pure CP ───────────────────────────────────────────────────────
            print("        CP ...", end=" ", flush=True)
            cp_ms, cp_t = run_cp(inst, tl)
            print(f"MS={cp_ms}  ({cp_t:.1f}s)")

            # ── Hybrid per fix_ratio (using best-of-K schedule) ───────────────
            hybrid = {}   # ratio -> (ms, time)
            for fr in FIX_RATIOS:
                print(f"        Hybrid fix={int(fr*100)}% ...", end=" ", flush=True)
                hms, ht = run_hybrid(inst, bok_sched, tl, fr)
                hybrid[fr] = (hms, ht)
                print(f"MS={hms}  ({ht:.1f}s)")

            # ── Build row ─────────────────────────────────────────────────────
            row = {
                "Benchmark"   : bm_name,
                "Instance"    : name,
                "Size"        : f"{N}x{M}",
                "Num_Jobs"    : N,
                "Num_Machines": M,
                "BKS"         : bks,
                # --- RL ---
                "RL_Greedy"   : greedy_ms,
                "RL_BoK"      : bok_ms,
                "RL_Time_s"   : round(rl_t, 3),
                "Gap_RL_Greedy": gap(greedy_ms, bks),
                "Gap_RL_BoK"  : gap(bok_ms, bks),
                # --- Pure CP ---
                "CP_Cold"     : cp_ms,
                "CP_Time_s"   : round(cp_t, 1),
                "Gap_CP"      : gap(cp_ms, bks),
            }

            for fr in FIX_RATIOS:
                hms, ht = hybrid[fr]
                pct = int(fr * 100)
                row[f"Hybrid_{pct}pct"]           = hms
                row[f"Hybrid_{pct}pct_Time_s"]    = round(ht, 1)
                row[f"Gap_Hybrid_{pct}pct"]        = gap(hms, bks)

            bm_rows.append(row)
            all_rows.append(row)

            # ── Save Incrementally after EVERY instance ───────────────────────
            pd.DataFrame(bm_rows).to_csv(out_bm, index=False)

            # Print mini summary line
            best_hybrid = min(hybrid.values(), key=lambda x: x[0])[0]
            print(f"        → RL={greedy_ms:.0f} | BoK={bok_ms:.0f} | CP={cp_ms} "
                  f"| BestHybrid={best_hybrid} | BKS={bks}")

        # ── Print summary table for this benchmark (after loop) ───────────────
        df_bm = pd.DataFrame(bm_rows)
        print(f"\n  → Saved {out_bm}")

        # ── Print summary table for this benchmark ────────────────────────────
        disp_cols = (["Instance", "Size", "BKS", "RL_Greedy", "RL_BoK", "CP_Cold"]
                     + [f"Hybrid_{int(fr*100)}pct" for fr in FIX_RATIOS]
                     + ["Gap_RL_BoK", "Gap_CP"]
                     + [f"Gap_Hybrid_{int(fr*100)}pct" for fr in FIX_RATIOS])
        print(f"\n  {'─'*60}")
        print(f"  SUMMARY — {bm_name}")
        print(f"  {'─'*60}")
        print(df_bm[[c for c in disp_cols if c in df_bm.columns]].to_string(index=False))

    # ── Save combined CSV ─────────────────────────────────────────────────────
    df_all = pd.DataFrame(all_rows)
    out_all = os.path.join(eval_dir, "full_benchmark_all.csv")
    df_all.to_csv(out_all, index=False)
    print(f"\n\n✅  All results saved to {out_all}")
    print(f"    Per-benchmark CSVs in {eval_dir}/full_benchmark_<name>.csv\n")

    # ── Grand summary ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("  GRAND SUMMARY — Mean Gap across all instances")
    print("=" * 70)
    gap_cols = (["Gap_RL_BoK", "Gap_CP"]
                + [f"Gap_Hybrid_{int(fr*100)}pct" for fr in FIX_RATIOS])
    numeric_gap = df_all[gap_cols].apply(pd.to_numeric, errors="coerce")
    means = numeric_gap.mean().round(1)
    for col, val in means.items():
        print(f"  {col:<28s}: {val}%")


if __name__ == "__main__":
    main()
