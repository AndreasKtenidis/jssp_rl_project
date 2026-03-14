"""
full_benchmark_eval.py (Enhanced Version with Advanced Hybrids)
========================================
- RL Greedy / Best-of-K (K=10)
- Pure CP (Cold Start) with Adaptive TL
- Hybrid CP (RL BoK Warm-Start)
- Advanced Hybrid Modes (Machine Precedence, CP Fixation, CP Hint)
- Records: Makespan, Time_to_Best, Status, Gap%, RL Times
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
from cp.cp_solver import solve_with_rl_warmstart, solve_instance_your_version, solve_hybrid_advanced

# ─── Configuration ──────────────────────────────────────────────────────────
BEST_OF_K    = 10
FIX_RATIOS   = [0.0, 0.1, 0.2, 0.3, 0.4]
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
    """Adaptive time limits."""
    ops = n * m
    if   ops <= 225:  return 45   # <= 15x15
    elif ops <= 600:  return 90   # <= 30x20
    else:             return 180  # Larger

# ─── RL Rollout helpers (Optimized) ─────────────────────────────────────────
# KEY INSIGHT from features.py / make_hetero_data:
#   STATIC (unchanging per instance): edge_index_dict, processing_time normalization
#   DYNAMIC (changes each step):      remaining_ops, machine_loads
# We pre-compute static parts once, then only rebuild the dynamic node features.

from torch_geometric.data import HeteroData

def _build_static_cache(env):
    """Pre-compute everything that doesn't change during a rollout."""
    # 1. Edge indices (depend only on machine assignments, never change)
    static_edges = HeteroGATv2.build_edge_index_dict(env.machines)
    job_edges  = static_edges[('op', 'job', 'op')].to(DEVICE)
    mach_edges = static_edges[('op', 'machine', 'op')].to(DEVICE)

    # 2. Processing time feature — normalize once (values never change)
    proc_time = env.times.flatten().to(torch.float32)
    proc_time_norm = (proc_time - proc_time.mean()) / (proc_time.std() + 1e-6)

    return job_edges, mach_edges, proc_time_norm


def _fast_hetero_data(env, job_edges, mach_edges, proc_time_norm):
    """Build HeteroData using cached static parts; only recompute dynamic features."""
    num_jobs, num_machines = env.num_jobs, env.num_machines

    # --- DYNAMIC: remaining_ops ---
    scheduled_cumsum = torch.cumsum(env.state, dim=1)
    scheduled_before = torch.cat([
        torch.zeros(num_jobs, 1, device=DEVICE),
        scheduled_cumsum[:, :-1]
    ], dim=1)
    remaining_ops = (num_machines - scheduled_before).flatten().to(torch.float32)
    remaining_ops_norm = (remaining_ops - remaining_ops.mean()) / (remaining_ops.std() + 1e-6)

    # --- DYNAMIC: machine_loads ---
    unscheduled = (env.state == 0).float()
    masked_times = env.times * unscheduled
    machine_loads = torch.zeros(num_machines, device=DEVICE)
    machine_loads.scatter_add_(0, env.machines.flatten(), masked_times.flatten())
    machine_loads_rep = machine_loads[env.machines.flatten()].to(torch.float32)
    machine_loads_norm = (machine_loads_rep - machine_loads_rep.mean()) / (machine_loads_rep.std() + 1e-6)

    # --- Combine (same order as prepare_features in features.py) ---
    x = torch.stack([proc_time_norm, remaining_ops_norm, machine_loads_norm], dim=1)

    data = HeteroData()
    data['op'].x = x
    data['op', 'job', 'op'].edge_index     = job_edges
    data['op', 'machine', 'op'].edge_index = mach_edges
    return data


@torch.no_grad()
def _fast_single_rollout(env, model, job_edges, mach_edges, proc_norm, stochastic: bool):
    """Rollout using pre-computed static cache. Returns (makespan, schedule, net_gnn_time)."""
    env.reset()
    done = False
    net_gnn_time = 0.0
    while not done:
        data = _fast_hetero_data(env, job_edges, mach_edges, proc_norm)
        t_gnn = time.time()
        logits, _ = model(data)
        net_gnn_time += time.time() - t_gnn

        avail = env.get_available_actions()
        if not avail:
            return float("inf"), [], net_gnn_time
        mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=DEVICE)
        mask[avail] = True
        logits = logits.masked_fill(~mask, -1e10)
        action = Categorical(logits=logits).sample().item() if stochastic else logits.argmax().item()
        _, _, done, _ = env.step(action)
    return env.get_makespan(), env.extract_job_assignments(), net_gnn_time


@torch.no_grad()
def run_rl(instance, model):
    times_in    = instance["times"].clone().detach().to(DEVICE) if torch.is_tensor(instance["times"])    else torch.tensor(instance["times"]).to(DEVICE)
    machines_in = instance["machines"].clone().detach().to(DEVICE) if torch.is_tensor(instance["machines"]) else torch.tensor(instance["machines"]).to(DEVICE)
    env = JSSPEnvironment(times_in, machines_in, device=DEVICE, use_shaping_rewards=False)

    # ── Build static cache ONCE per instance ──────────────────────────────────
    job_edges, mach_edges, proc_norm = _build_static_cache(env)

    # ── Greedy ────────────────────────────────────────────────────────────────
    t0 = time.time()
    greedy_ms, greedy_sched, greedy_gnn_t = _fast_single_rollout(env, model, job_edges, mach_edges, proc_norm, stochastic=False)
    greedy_wall_t = time.time() - t0

    # ── Best-of-K ─────────────────────────────────────────────────────────────
    t1 = time.time()
    bok_ms, bok_sched, bok_gnn_t = greedy_ms, greedy_sched, greedy_gnn_t
    for _ in range(BEST_OF_K):
        ms, sched, gnn_t = _fast_single_rollout(env, model, job_edges, mach_edges, proc_norm, stochastic=True)
        if ms < bok_ms:
            bok_ms, bok_sched, bok_gnn_t = ms, sched, gnn_t
    bok_wall_t = time.time() - t1

    # Returns: greedy_ms, greedy_wall_t, bok_ms, bok_wall_t, bok_sched
    return greedy_ms, greedy_wall_t, bok_ms, bok_wall_t, bok_sched

def run_cp(instance, tl, bks):
    times, machines = instance["times"], instance["machines"]
    t_np = times.numpy() if hasattr(times, "numpy") else times
    m_np = machines.numpy() if hasattr(machines, "numpy") else machines
    ms, _, status, t_best = solve_instance_your_version(t_np, m_np, time_limit_s=tl, bks=bks)
    return ms, status, t_best

def run_hybrid(instance, schedule, tl, fix_ratio, bks):
    times, machines = instance["times"], instance["machines"]
    t_np = times.numpy() if hasattr(times, "numpy") else times
    m_np = machines.numpy() if hasattr(machines, "numpy") else machines
    ms, _, status, t_best = solve_with_rl_warmstart(t_np, m_np, rl_schedule=schedule, time_limit_s=tl, fix_ratio=fix_ratio, bks=bks)
    return ms, status, t_best

def run_hybrid_adv(instance, schedule, tl, bks, mode, fix_ratio=0.0):
    times, machines = instance["times"], instance["machines"]
    t_np = times.numpy() if hasattr(times, "numpy") else times
    m_np = machines.numpy() if hasattr(machines, "numpy") else machines
    ms, _, status, t_best = solve_hybrid_advanced(t_np, m_np, rl_schedule=schedule, mode=mode, fix_ratio=fix_ratio, time_limit_s=tl, bks=bks)
    return ms, status, t_best

def gap(ms, bks):
    if bks <= 0 or ms is None or ms == float('inf'): return None
    return round((ms - bks) / bks * 100, 2)

def main():
    saved_dir = os.path.join(BASE_DIR, "saved")
    eval_dir  = os.path.join(BASE_DIR, "eval", "results_v2_callback")
    model_dir = os.path.join(BASE_DIR, "outputs", "models")
    os.makedirs(eval_dir, exist_ok=True)

    ckpt_path = None
    for cand in ["best_ppo_latest.pt", "best_ppo_Phase_8.pt", "best_ppo_Phase_7.pt"]:
        p = os.path.join(model_dir, cand)
        if os.path.exists(p): ckpt_path = p; break

    if not ckpt_path: return print("[ERROR] No model found.")
    print(f"✅ Loading: {os.path.basename(ckpt_path)}")
    model = ActorCriticPPO(3, 64, 32, 32, 32).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
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
        with open(path, "rb") as f: instances = pickle.load(f)

        out_bm = os.path.join(eval_dir, f"full_benchmark_adv_{bm_name.lower()}.csv")
        existing_instances = set()
        if os.path.exists(out_bm):
            try:
                df_ex = pd.read_csv(out_bm)
                existing_instances = set(df_ex["Instance"].tolist())
                bm_rows = df_ex.to_dict("records")
                print(f"[Resume] {bm_name}: Skipping {len(existing_instances)}")
            except: bm_rows = []
        else: bm_rows = []

        for i, inst in enumerate(instances):
            name = inst.get("name", f"{bm_name}_{i}")
            if name in existing_instances: continue

            N, M = (inst["times"].shape if hasattr(inst["times"], "shape") else (len(inst["times"]), len(inst["times"][0])))
            bks = inst.get("bks", -1)
            tl = cp_time_limit(N, M)
            print(f"\n[{i+1:3d}] {name} ({N}x{M}) BKS={bks} TL={tl}s", flush=True)

            # RL
            g_ms, g_t, b_ms, b_t, b_sched = run_rl(inst, model)
            print(f"  RL: Greedy={g_ms} ({g_t:.2f}s) Gap={gap(g_ms, bks)}% | BoK={b_ms} ({b_t:.2f}s) Gap={gap(b_ms, bks)}%", flush=True)

            # CP Cold
            cp_ms, cp_status, cp_t_best = run_cp(inst, tl, bks)
            print(f"  CP: MS={cp_ms} ({cp_t_best:.1f}s) Gap={gap(cp_ms, bks)}% Status={cp_status}", flush=True)

            # Hybrid Standard (0-40%)
            row_data = {
                "Benchmark": bm_name, "Instance": name, "Size": f"{N}x{M}", "BKS": bks,
                "RL_Greedy_MS": g_ms, "RL_Greedy_T": round(g_t, 3), "Gap_RL_Greedy": gap(g_ms, bks),
                "RL_BoK_MS": b_ms, "RL_BoK_T": round(b_t, 3), "Gap_RL_BoK": gap(b_ms, bks),
                "CP_MS": cp_ms, "CP_T_Best": round(cp_t_best, 1), "CP_Status": cp_status, "Gap_CP": gap(cp_ms, bks)
            }
            
            for fr in [0.0, 0.1, 0.2, 0.3, 0.4]:
                h_ms, h_status, h_t_best = run_hybrid(inst, b_sched, tl, fr, bks)
                pct = int(fr*100)
                row_data[f"Hybrid_{pct}pct_MS"] = h_ms
                row_data[f"Hybrid_{pct}pct_T_Best"] = round(h_t_best, 1)
                row_data[f"Hybrid_{pct}pct_Status"] = h_status
                row_data[f"Gap_Hybrid_{pct}pct"] = gap(h_ms, bks)
                print(f"  Hybrid {pct}%: MS={h_ms} ({h_t_best:.1f}s) Gap={gap(h_ms, bks)}% Status={h_status}", flush=True)

            # Advanced Hybrids
            # 1. Machine Precedence
            m_ms, m_status, m_t_best = run_hybrid_adv(inst, b_sched, tl, bks, 'machine_precedence')
            row_data["Hybrid_MachPrec_MS"] = m_ms
            row_data["Hybrid_MachPrec_T_Best"] = round(m_t_best, 1)
            row_data["Hybrid_MachPrec_Status"] = m_status
            row_data["Gap_Hybrid_MachPrec"] = gap(m_ms, bks)
            print(f"  Hybrid MachPrec: MS={m_ms} ({m_t_best:.1f}s) Gap={gap(m_ms, bks)}% Status={m_status}", flush=True)

            # 2. CP Partial Fix (20% & 50%)
            for fr_cp in [0.2, 0.5]:
                p_ms, p_status, p_t_best = run_hybrid_adv(inst, b_sched, tl, bks, 'cp_fixation', fix_ratio=fr_cp)
                pct_cp = int(fr_cp*100)
                row_data[f"Hybrid_CPFix_{pct_cp}pct_MS"] = p_ms
                row_data[f"Hybrid_CPFix_{pct_cp}pct_T_Best"] = round(p_t_best, 1)
                row_data[f"Hybrid_CPFix_{pct_cp}pct_Status"] = p_status
                row_data[f"Gap_Hybrid_CPFix_{pct_cp}pct"] = gap(p_ms, bks)
                print(f"  Hybrid CPFix {pct_cp}%: MS={p_ms} ({p_t_best:.1f}s) Gap={gap(p_ms, bks)}% Status={p_status}", flush=True)

            # 3. CP Super Hint
            sh_ms, sh_status, sh_t_best = run_hybrid_adv(inst, b_sched, tl, bks, 'cp_hint')
            row_data["Hybrid_CPHint_MS"] = sh_ms
            row_data["Hybrid_CPHint_T_Best"] = round(sh_t_best, 1)
            row_data["Hybrid_CPHint_Status"] = sh_status
            row_data["Gap_Hybrid_CPHint"] = gap(sh_ms, bks)
            print(f"  Hybrid CPHint: MS={sh_ms} ({sh_t_best:.1f}s) Gap={gap(sh_ms, bks)}% Status={sh_status}", flush=True)

            bm_rows.append(row_data); all_rows.append(row_data)
            pd.DataFrame(bm_rows).to_csv(out_bm, index=False)

    # Summary file update
    final_all_path = os.path.join(eval_dir, "full_benchmark_adv_all.csv")
    if os.path.exists(final_all_path) and only_bench:
        # If we ran only one bench, append to existing or merge
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
        print(f"\n✅ Final advanced results saved to {final_all_path}")

if __name__ == "__main__": main()
