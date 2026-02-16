# jssp_rl/cp/cp_solver.py
from ortools.sat.python import cp_model
from utils.logging_utils import plot_gantt_chart
import os
import numpy as np
import torch

def _to_py_int(x):
    # Safely convert torch/numpy scalars to Python int
    if isinstance(x, torch.Tensor):
        return int(x.item())
    if isinstance(x, (np.integer, np.floating)):
        return int(x)
    return int(x)

def solve_instance_your_version(times, machines, time_limit_s: float = 10.0):
    """
    Solve one JSSP instance with CP-SAT.
    Args:
        times: [J, M] torch.Tensor or np.ndarray or list of lists
        machines: [J, M] same as above; machine IDs must be 0..M-1
        time_limit_s: CP-SAT wall time in seconds
    Returns:
        (makespan:int, schedule:list[dict]) or (None, [])
        schedule entries match env.extract_job_assignments() keys.
    """
    # Normalize inputs to numpy for indexing speed and type safety
    if isinstance(times, torch.Tensor):
        times_np = times.detach().cpu().numpy()
    elif isinstance(times, np.ndarray):
        times_np = times
    else:
        times_np = np.asarray(times, dtype=np.int64)

    if isinstance(machines, torch.Tensor):
        machines_np = machines.detach().cpu().numpy()
    elif isinstance(machines, np.ndarray):
        machines_np = machines
    else:
        machines_np = np.asarray(machines, dtype=np.int64)

    num_jobs, num_machines = times_np.shape
    model = cp_model.CpModel()

    jobs_starts, jobs_ends, intervals = {}, {}, {}

    horizon = int(times_np.sum())  # safe upper bound
    for j in range(num_jobs):
        for o in range(num_machines):
            dur = _to_py_int(times_np[j, o])
            start = model.NewIntVar(0, horizon, f'start_{j}_{o}')
            end = model.NewIntVar(0, horizon, f'end_{j}_{o}')
            interval = model.NewIntervalVar(start, dur, end, f'interval_{j}_{o}')
            jobs_starts[(j, o)] = start
            jobs_ends[(j, o)] = end
            intervals[(j, o)] = interval

    # Precedence per job
    for j in range(num_jobs):
        for o in range(1, num_machines):
            model.Add(jobs_starts[(j, o)] >= jobs_ends[(j, o - 1)])

    # No overlap per machine
    machine_intervals = {}
    for j in range(num_jobs):
        for o in range(num_machines):
            m = _to_py_int(machines_np[j, o])
            machine_intervals.setdefault(m, []).append(intervals[(j, o)])
    for m, ints in machine_intervals.items():
        model.AddNoOverlap(ints)

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, [jobs_ends[(j, num_machines - 1)] for j in range(num_jobs)])
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    # Optional: a bit more deterministic behavior
    solver.parameters.num_search_workers = 8  # adjust to your CPU
    # solver.parameters.random_seed = 0

    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = []
        for j in range(num_jobs):
            for o in range(num_machines):
                start = solver.Value(jobs_starts[(j, o)])
                end = solver.Value(jobs_ends[(j, o)])
                m = _to_py_int(machines_np[j, o])
                schedule.append({
                    "job_id": j,
                    "operation_index": o,
                    "machine": m,
                    "start_time": start,
                    "end_time": end
                })
        return solver.ObjectiveValue(), schedule

    return None, []

def solve_with_rl_warmstart(times, machines, rl_schedule, time_limit_s: float = 60.0, fix_ratio: float = 0.0):
    """
    Solve JSSP using RL schedule as a warm start.
    Args:
        rl_schedule: List[dict] from env.extract_job_assignments()
        fix_ratio: 0.0 to 1.0. If > 0, 'hard fixes' this ratio of operations to RL start times.
    """
    if isinstance(times, torch.Tensor): times_np = times.detach().cpu().numpy()
    else: times_np = np.asarray(times)
    
    if isinstance(machines, torch.Tensor): machines_np = machines.detach().cpu().numpy()
    else: machines_np = np.asarray(machines)

    num_jobs, num_machines = times_np.shape
    model = cp_model.CpModel()
    
    jobs_starts, jobs_ends, intervals = {}, {}, {}
    horizon = int(times_np.sum())

    # 1. Variables and Intervals
    for j in range(num_jobs):
        for o in range(num_machines):
            dur = _to_py_int(times_np[j, o])
            start = model.NewIntVar(0, horizon, f's_{j}_{o}')
            end = model.NewIntVar(0, horizon, f'e_{j}_{o}')
            interval = model.NewIntervalVar(start, dur, end, f'i_{j}_{o}')
            jobs_starts[(j, o)] = start
            jobs_ends[(j, o)] = end
            intervals[(j, o)] = interval

    # 2. JSSP Constraints
    for j in range(num_jobs):
        for o in range(1, num_machines):
            model.Add(jobs_starts[(j, o)] >= jobs_ends[(j, o-1)])
    
    machine_intervals = {}
    for j in range(num_jobs):
        for o in range(num_machines):
            m = _to_py_int(machines_np[j, o])
            machine_intervals.setdefault(m, []).append(intervals[(j, o)])
    for m, ints in machine_intervals.items():
        model.AddNoOverlap(ints)

    # 3. RL Warm Start / Hints
    rl_makespan = 0
    if rl_schedule:
        # Sort RL schedule to identify "early" operations if we want to fix a ratio
        rl_schedule_sorted = sorted(rl_schedule, key=lambda x: x['start_time'])
        num_to_fix = int(len(rl_schedule_sorted) * fix_ratio)
        
        for i, entry in enumerate(rl_schedule_sorted):
            j, o = entry['job_id'], entry['operation_index']
            rl_start = int(entry['start_time'])
            rl_end = int(entry['end_time'])
            
            if i < num_to_fix:
                # Hard fix: the first few ops must start exactly at RL times
                model.Add(jobs_starts[(j, o)] == rl_start)
            else:
                # Soft hint: suggestion to the solver
                model.AddHint(jobs_starts[(j, o)], rl_start)
            
            rl_makespan = max(rl_makespan, rl_end)

    # 4. Objective with RL Upper Bound
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, [jobs_ends[(j, num_machines - 1)] for j in range(num_jobs)])
    
    if rl_makespan > 0:
        # CP should at least do as well as RL
        model.Add(makespan <= int(rl_makespan))
    
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8 
    
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        final_schedule = []
        for j in range(num_jobs):
            for o in range(num_machines):
                final_schedule.append({
                    "job_id": j,
                    "operation_index": o,
                    "machine": _to_py_int(machines_np[j, o]),
                    "start_time": solver.Value(jobs_starts[(j, o)]),
                    "end_time": solver.Value(jobs_ends[(j, o)])
                })
        return solver.ObjectiveValue(), final_schedule

    return None, []

def run_cp_on_all(instances, save_gantt_dir=None, time_limit_s: float = 10.0):
    results = []
    for i, inst in enumerate(instances):
        print(f"🧠 Solving CP on instance {i+1}/{len(instances)}")

        ms, schedule = solve_instance_your_version(inst["times"], inst["machines"], time_limit_s=time_limit_s)
        results.append({"instance_id": i, "cp_makespan": ms})

        if schedule and save_gantt_dir:
            save_path = os.path.join(save_gantt_dir, f"gantt_cp_taillardV2_{i:02d}.png")
            plot_gantt_chart(schedule, save_path=save_path, show_op_index=True)

    return results
