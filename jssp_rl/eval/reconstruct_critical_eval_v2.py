"""
reconstruct_critical_eval_v2.py
==========================================================
Critical-Path Large Neighbourhood Search (CP-LNS) Evaluator - V2

Strategy V2 (90% Reconstruct / 10% Fixed):
 1. Get RL Best-of-K solution as starting point.
 2. Select 10% of total operations dynamically as "easy" (smallest processing time).
 3. Iteratively extract Critical Paths from the RL schedule until
    the total reconstructed set reaches 80% coverage (10% initial easy + 70% CP).
 4. Take another 10% of total operations as "easy" from the remaining 20%.
 5. The "Reconstruct Set" = 90% total.
 6. Pass to CP-SAT: Reconstruct Set is free, rest is FIXED.

Output format matches full_benchmark_eval.py.
"""

import os, sys, pickle, time
import torch
import numpy as np
import pandas as pd
from torch.distributions import Categorical
from ortools.sat.python import cp_model

# ─── Path setup ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from models.Hetero_actor_critic_ppo import ActorCriticPPO
from models.gin import HeteroGATv2
from env.jssp_environment import JSSPEnvironment
from utils.features import make_hetero_data
from cp.cp_solver import JSSPSolutionCallback, _to_py_int, solve_instance_your_version

from torch_geometric.data import HeteroData

# ─── Configuration ───────────────────────────────────────────────────────────
BEST_OF_K = 10
INITIAL_EASY_RATIO = 0.10
TARGET_CP_PLUS_EASY = 0.80
FINAL_EASY_RATIO = 0.10
MODE_LABEL = "V2_90pct"

BENCHMARKS = {
    "FT":       "benchmark_ft.pkl",
    "Taillard": "benchmark_taillard.pkl",
    "DMU":      "benchmark_dmu.pkl",
    "ABZ":      "benchmark_abz.pkl",
    "Lawrence": "benchmark_la.pkl",
    "ORB":      "benchmark_orb.pkl",
    "SWV":      "benchmark_swv.pkl",
    "YN":       "benchmark_yn.pkl",
    "DacolTeppan2022": "benchmark_DacolTeppan2022.pkl",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cp_time_limit(n, m):
    ops = n * m
    if   ops <= 225: return 45
    elif ops <= 600: return 90
    else:            return 180


# ─── RL Rollout (same fast-cached approach as full_benchmark_eval.py) ────────
def _build_static_cache(env):
    static_edges = HeteroGATv2.build_edge_index_dict(env.machines)
    job_edges  = static_edges[('op', 'job',     'op')].to(DEVICE)
    mach_edges = static_edges[('op', 'machine', 'op')].to(DEVICE)
    proc_time  = env.times.flatten().to(torch.float32)
    proc_norm  = (proc_time - proc_time.mean()) / (proc_time.std() + 1e-6)
    return job_edges, mach_edges, proc_norm


def _fast_hetero_data(env, job_edges, mach_edges, proc_norm):
    num_jobs, num_machines = env.num_jobs, env.num_machines
    scheduled_cumsum = torch.cumsum(env.state, dim=1)
    scheduled_before = torch.cat([
        torch.zeros(num_jobs, 1, device=DEVICE),
        scheduled_cumsum[:, :-1]
    ], dim=1)
    remaining_ops      = (num_machines - scheduled_before).flatten().to(torch.float32)
    remaining_ops_norm = (remaining_ops - remaining_ops.mean()) / (remaining_ops.std() + 1e-6)

    unscheduled   = (env.state == 0).float()
    masked_times  = env.times * unscheduled
    machine_loads = torch.zeros(num_machines, device=DEVICE)
    machine_loads.scatter_add_(0, env.machines.flatten(), masked_times.flatten())
    ml_rep       = machine_loads[env.machines.flatten()].to(torch.float32)
    ml_norm      = (ml_rep - ml_rep.mean()) / (ml_rep.std() + 1e-6)

    x = torch.stack([proc_norm, remaining_ops_norm, ml_norm], dim=1)
    data = HeteroData()
    data['op'].x = x
    data['op', 'job',     'op'].edge_index = job_edges
    data['op', 'machine', 'op'].edge_index = mach_edges
    return data


@torch.no_grad()
def _fast_single_rollout(env, model, job_edges, mach_edges, proc_norm, stochastic: bool):
    env.reset()
    done = False
    while not done:
        data   = _fast_hetero_data(env, job_edges, mach_edges, proc_norm)
        logits, _ = model(data)
        avail  = env.get_available_actions()
        if not avail:
            return float("inf"), []
        mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=DEVICE)
        mask[avail] = True
        logits = logits.masked_fill(~mask, -1e10)
        action = Categorical(logits=logits).sample().item() if stochastic else logits.argmax().item()
        _, _, done, _ = env.step(action)
    return env.get_makespan(), env.extract_job_assignments()


@torch.no_grad()
def run_rl(instance, model):
    times_in    = (instance["times"].clone().detach().to(DEVICE)
                   if torch.is_tensor(instance["times"])
                   else torch.tensor(instance["times"]).to(DEVICE))
    machines_in = (instance["machines"].clone().detach().to(DEVICE)
                   if torch.is_tensor(instance["machines"])
                   else torch.tensor(instance["machines"]).to(DEVICE))
    env = JSSPEnvironment(times_in, machines_in, device=DEVICE, use_shaping_rewards=False)
    job_edges, mach_edges, proc_norm = _build_static_cache(env)

    # Greedy
    t0 = time.time()
    greedy_ms, _ = _fast_single_rollout(env, model, job_edges, mach_edges, proc_norm, stochastic=False)
    greedy_t = time.time() - t0

    # Best-of-K
    t1       = time.time()
    bok_ms   = greedy_ms
    bok_sched = []
    for _ in range(BEST_OF_K):
        ms, sched = _fast_single_rollout(env, model, job_edges, mach_edges, proc_norm, stochastic=True)
        if ms < bok_ms:
            bok_ms, bok_sched = ms, sched
    if not bok_sched:
        _, bok_sched = _fast_single_rollout(env, model, job_edges, mach_edges, proc_norm, stochastic=False)
    bok_t = time.time() - t1

    return greedy_ms, greedy_t, bok_ms, bok_t, bok_sched


# ─── Critical-Path Extraction ────────────────────────────────────────────────

def find_critical_path_from_schedule(schedule, excluded=None):
    """
    Extract one critical path (traceback from makespan) from `schedule`.
    `excluded`: set of (job_id, op_index) to ignore (already selected in prev iterations).
    Returns: set of (job_id, op_index) on this critical path.
    """
    if not schedule:
        return set()

    excluded = excluded or set()
    active   = [e for e in schedule if (e['job_id'], e['operation_index']) not in excluded]
    if not active:
        return set()

    makespan = max(e['end_time'] for e in active)
    lookup   = {(e['job_id'], e['operation_index']): e for e in active}

    machine_map = {}
    for entry in active:
        machine_map.setdefault(entry['machine'], []).append(entry)
    
    op_to_m_idx = {}
    for m in machine_map:
        machine_map[m] = sorted(machine_map[m], key=lambda x: x['start_time'])
        for idx, entry in enumerate(machine_map[m]):
            op_to_m_idx[(entry['job_id'], entry['operation_index'])] = idx

    cp_ops  = set()
    visited = set()
    stack   = []

    for entry in active:
        if entry['end_time'] == makespan:
            stack.append(entry)

    while stack:
        curr = stack.pop()
        key = (curr['job_id'], curr['operation_index'])
        if key in visited or key in excluded:
            continue
        visited.add(key)
        cp_ops.add(key)
        
        s = curr['start_time']
        if s == 0:
            continue
            
        # Job predecessor
        if curr['operation_index'] > 0:
            prev_j = lookup.get((curr['job_id'], curr['operation_index'] - 1))
            if prev_j and prev_j['end_time'] == s:
                stack.append(prev_j)
                
        # Machine predecessor
        idx = op_to_m_idx.get(key)
        if idx is not None and idx > 0:
            prev_m = machine_map[curr['machine']][idx - 1]
            if prev_m['end_time'] == s:
                stack.append(prev_m)

    return cp_ops


def select_reconstruct_ops(schedule, times_np, machines_np):
    """
    V2 Strategy:
    1. Initial 10% easy operations (smallest processing time).
    2. Iteratively extract critical paths until reconstruct_set has 80% of total ops.
    3. Add 10% final easy operations from the remaining operations.
    
    Returns:
        reconstruct_set : set of (job_id, op_index) that CP is free to change
        fixed_set       : all others (hard-fixed to RL start times)
    """
    num_jobs, num_machines = times_np.shape
    total_ops              = num_jobs * num_machines
    
    # Sort all operations by processing time
    all_ops_list = [(j, o) for j in range(num_jobs) for o in range(num_machines)]
    all_ops_list.sort(key=lambda op: int(times_np[op[0], op[1]]))
    
    # 1. Initial 10% easy ops
    initial_easy_count = int(total_ops * INITIAL_EASY_RATIO)
    selected_initial_easy = set(all_ops_list[:initial_easy_count])
    
    reconstruct_set = set(selected_initial_easy)
    
    # 2. Iterative critical paths
    target_cp_plus_easy_count = int(total_ops * TARGET_CP_PLUS_EASY)
    
    while len(reconstruct_set) < target_cp_plus_easy_count:
        cp_path = find_critical_path_from_schedule(schedule, excluded=reconstruct_set)
        if not cp_path:
            break
        reconstruct_set.update(cp_path)
        
    # 3. Final 10% easy ops from the remaining
    remaining_list = [op for op in all_ops_list if op not in reconstruct_set]
    final_easy_count = int(total_ops * FINAL_EASY_RATIO)
    selected_final_easy = set(remaining_list[:final_easy_count])
    
    reconstruct_set.update(selected_final_easy)
    
    fixed_set = set(all_ops_list) - reconstruct_set
    return reconstruct_set, fixed_set


# ─── CP-LNS Solver ──────────────────────────────────────────────────────────

def solve_cp_lns(times, machines, rl_schedule, reconstruct_set, time_limit_s, bks=None):
    """
    CP-SAT solver using Critical-Path LNS:
    - Operations in `reconstruct_set`: FREE (only hint from RL)
    - Operations NOT in `reconstruct_set` (fixed_set): HARD FIXED to RL start times
    """
    if isinstance(times, torch.Tensor):
        times_np = times.detach().cpu().numpy()
    else:
        times_np = np.asarray(times)
    if isinstance(machines, torch.Tensor):
        machines_np = machines.detach().cpu().numpy()
    else:
        machines_np = np.asarray(machines)

    num_jobs, num_machines = times_np.shape
    model   = cp_model.CpModel()
    horizon = int(times_np.sum())

    jobs_starts, jobs_ends, intervals = {}, {}, {}

    for j in range(num_jobs):
        for o in range(num_machines):
            dur      = _to_py_int(times_np[j, o])
            start    = model.NewIntVar(0, horizon, f's_{j}_{o}')
            end      = model.NewIntVar(0, horizon, f'e_{j}_{o}')
            interval = model.NewIntervalVar(start, dur, end, f'i_{j}_{o}')
            jobs_starts[(j, o)] = start
            jobs_ends[(j, o)]   = end
            intervals[(j, o)]   = interval

    # Job precedence
    for j in range(num_jobs):
        for o in range(1, num_machines):
            model.Add(jobs_starts[(j, o)] >= jobs_ends[(j, o - 1)])

    # Machine no-overlap
    machine_intervals = {}
    for j in range(num_jobs):
        for o in range(num_machines):
            m = _to_py_int(machines_np[j, o])
            machine_intervals.setdefault(m, []).append(intervals[(j, o)])
    for m, ints in machine_intervals.items():
        model.AddNoOverlap(ints)

    # Apply RL start times: FIXED or HINT
    rl_lookup   = {(e['job_id'], e['operation_index']): e for e in rl_schedule}
    rl_makespan = max(e['end_time'] for e in rl_schedule)

    for (j, o), entry in rl_lookup.items():
        rl_start = int(entry['start_time'])
        if (j, o) in reconstruct_set:
            model.AddHint(jobs_starts[(j, o)], rl_start)   # FREE, but hinted
        else:
            model.Add(jobs_starts[(j, o)] == rl_start)     # HARD FIXED

    # Objective: minimize makespan, upper bounded by RL makespan
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, [jobs_ends[(j, num_machines - 1)] for j in range(num_jobs)])
    model.Add(makespan <= int(rl_makespan))
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers  = 8

    cb          = JSSPSolutionCallback(makespan, bks=bks)
    status_code = solver.Solve(model, cb)

    final_status = "FEASIBLE"
    if status_code == cp_model.OPTIMAL:
        final_status = "OPTIMAL"
    elif cb.status == "BKS_REACHED":
        final_status = "BKS_REACHED"
    elif cb.status == "BKS_BEATEN":
        final_status = "BKS_BEATEN"
    elif status_code == cp_model.INFEASIBLE:
        final_status = "INFEASIBLE"

    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver.ObjectiveValue(), final_status, cb.time_to_best

    return None, "FAILED", 0.0


# ─── Gap Utility ─────────────────────────────────────────────────────────────
def gap(ms, bks):
    if bks is None or bks <= 0 or ms is None or ms == float('inf'):
        return None
    return round((ms - bks) / bks * 100, 2)


# ─── Main evaluation loop ────────────────────────────────────────────────────
def main():
    saved_dir = os.path.join(BASE_DIR, "saved")
    eval_dir  = os.path.join(BASE_DIR, "eval", "results_V3_percentage_critical")
    model_dir = os.path.join(BASE_DIR, "outputs", "models")
    os.makedirs(eval_dir, exist_ok=True)

    # Load model
    ckpt_path = None
    for cand in ["best_ppo_latest.pt", "best_ppo_Phase_8.pt", "best_ppo_Phase_7.pt"]:
        p = os.path.join(model_dir, cand)
        if os.path.exists(p):
            ckpt_path = p
            break
    if not ckpt_path:
        return print("[ERROR] No model checkpoint found.")

    print(f"Loading: {os.path.basename(ckpt_path)}")
    model = ActorCriticPPO(3, 64, 32, 32, 32).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    all_rows = []

    # Allow filtering via ENV
    only_bench = os.environ.get("BENCH_ONLY")

    for bm_name, bm_file in BENCHMARKS.items():
        if only_bench and bm_name != only_bench:
            continue

        path = os.path.join(saved_dir, bm_file)
        if not os.path.exists(path):
            print(f"[SKIP] {bm_file} not found.")
            continue

        with open(path, "rb") as f:
            instances = pickle.load(f)

        out_file = os.path.join(eval_dir, f"cplns_v2_{bm_name.lower()}.csv")

        # Resume support
        existing = set()
        bm_rows  = []
        if os.path.exists(out_file):
            try:
                df_ex    = pd.read_csv(out_file)
                existing = set(df_ex["Instance"].tolist())
                bm_rows  = df_ex.to_dict("records")
                print(f"[Resume] {bm_name}: Skipping {len(existing)} already done.")
            except Exception:
                bm_rows = []

        for i, inst in enumerate(instances):
            name = inst.get("name", f"{bm_name}_{i}")
            if name in existing:
                continue

            times_in    = inst["times"]
            machines_in = inst["machines"]
            times_np    = (times_in.numpy() if torch.is_tensor(times_in)
                           else np.asarray(times_in))
            machines_np = (machines_in.numpy() if torch.is_tensor(machines_in)
                           else np.asarray(machines_in))

            N, M = times_np.shape
            
            # Skip massive intractable instances due to RL computational bottleneck
            if N * M > 2000:
                print(f"\n[{i+1:3d}] {name} ({N}x{M}) skipped (N*M > 2,000)", flush=True)
                continue
                
            bks  = inst.get("bks", -1)
            tl   = cp_time_limit(N, M)

            print(f"\n[{i+1:3d}] {name} ({N}x{M}) BKS={bks} TL={tl}s", flush=True)

            # ── RL inference ──────────────────────────────────────────────────
            g_ms, g_t, b_ms, b_t, b_sched = run_rl(inst, model)
            print(f"  RL: Greedy={g_ms} ({g_t:.2f}s) Gap={gap(g_ms, bks)}% | "
                  f"BoK={b_ms} ({b_t:.2f}s) Gap={gap(b_ms, bks)}%", flush=True)

            # ── Pure CP baseline (cold start) ─────────────────────────────────
            cp_ms, _, cp_status, cp_t_best = solve_instance_your_version(
                times_np, machines_np, time_limit_s=tl,
                bks=bks if bks > 0 else None
            )
            print(f"  CP: MS={cp_ms} ({cp_t_best:.1f}s) Gap={gap(cp_ms, bks)}% Status={cp_status}", flush=True)

            row_data = {
                "Benchmark":    bm_name,
                "Instance":     name,
                "Size":         f"{N}x{M}",
                "BKS":          bks,
                "RL_Greedy_MS": g_ms,
                "RL_Greedy_T":  round(g_t, 3),
                "Gap_RL_Greedy": gap(g_ms, bks),
                "RL_BoK_MS":    b_ms,
                "RL_BoK_T":     round(b_t, 3),
                "Gap_RL_BoK":   gap(b_ms, bks),
                "CP_MS":        cp_ms,
                "CP_T_Best":    round(cp_t_best, 1),
                "CP_Status":    cp_status,
                "Gap_CP":       gap(cp_ms, bks),
            }

            # ── CP-LNS experiments ────────────────────────────────────────────
            reconstruct_set, fixed_set = select_reconstruct_ops(b_sched, times_np, machines_np)
            rc_count = len(reconstruct_set)
            fx_count = len(fixed_set)

            label = MODE_LABEL
            ms, status, t_best = solve_cp_lns(
                times_np, machines_np, b_sched,
                reconstruct_set=reconstruct_set,
                time_limit_s=tl,
                bks=bks if bks > 0 else None
            )
            g = gap(ms, bks)
            
            initial_easy_pct = int(INITIAL_EASY_RATIO * 100)
            final_easy_pct = int(FINAL_EASY_RATIO * 100)
            cp_plus_initial = int(TARGET_CP_PLUS_EASY * 100)
            critical_added = cp_plus_initial - initial_easy_pct
            total_free_pct = cp_plus_initial + final_easy_pct
            
            print(
                f"  {label} [InitialEasy={initial_easy_pct}%+CP={critical_added}%+FinalEasy={final_easy_pct}%={total_free_pct}% free, "
                f"Fixed={100-total_free_pct}%, ops: recon={rc_count} fix={fx_count}]: "
                f"MS={ms} ({t_best:.1f}s) Gap={g}% Status={status}",
                flush=True
            )

            row_data[f"CPLNS_{label}_MS"]     = ms
            row_data[f"CPLNS_{label}_T_Best"]  = round(t_best, 1)
            row_data[f"CPLNS_{label}_Status"]  = status
            row_data[f"Gap_CPLNS_{label}"]     = g

            bm_rows.append(row_data)
            all_rows.append(row_data)

            pd.DataFrame(bm_rows).to_csv(out_file, index=False)
            

    # Summary file update
    final_all_path = os.path.join(eval_dir, "cplns_v2_all.csv")
    if os.path.exists(final_all_path) and only_bench:
        try:
            df_existing = pd.read_csv(final_all_path)
            # Remove existing rows for this benchmark to avoid duplicates
            df_existing = df_existing[df_existing["Benchmark"] != only_bench]
            df_all = pd.concat([df_existing, pd.DataFrame(all_rows)], ignore_index=True)
            df_all.to_csv(final_all_path, index=False)
            print(f"\n✅ Updated summary in {final_all_path}")
        except:
            pd.DataFrame(all_rows).to_csv(final_all_path, index=False)
    else:
        pd.DataFrame(all_rows).to_csv(final_all_path, index=False)
        print(f"\n✅ Final results saved to {final_all_path}")


if __name__ == "__main__":
    main()
