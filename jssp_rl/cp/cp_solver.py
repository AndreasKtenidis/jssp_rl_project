# jssp_rl/cp/cp_solver.py
"""
Job Shop Scheduling Problem (JSSP) - CP-SAT Formulation
=======================================================

Problem Description:
-------------------
The Job Shop Scheduling Problem (JSSP) involves scheduling a set of jobs on a set of machines. 
Each job consists of a sequence of operations that must be performed in a predefined order. 
Each operation requires a specific machine for a fixed duration. The goal is to determine the 
start times of all operations to minimize the total completion time (makespan).

Mathematical Formulation:
-------------------------

1. Sets and Indices:
   - J: Set of jobs {1, ..., n}, indexed by i.
   - M: Set of machines {1, ..., m}, indexed by j.
   - O_i,k: The k-th operation of job i.

2. Parameters:
   - p_i,k: Processing time (duration) of operation O_i,k.
   - m_i,k: The machine required for operation O_i,k.

3. Decision Variables:
   - s_i,k: Start time of operation O_i,k (s_i,k >= 0).
   - C_max: The makespan (total completion time).

4. Constraints:
   - Precedence Constraints: Within each job, an operation cannot start until the previous 
     operation in its sequence is completed:
     s_i,k + p_i,k <= s_i,k+1  (for k = 1, ..., m-1)

   - Resource Constraints (No-Overlap): A machine can process only one operation at a time. 
     For any two operations O_i,k and O_i',k' assigned to the same machine (m_i,k = m_i',k'):
     (s_i,k + p_i,k <= s_i',k') OR (s_i',k' + p_i',k' <= s_i,k)

   - Makespan Definition: The makespan must be at least as large as the completion time 
     of every job's last operation:
     C_max >= s_i,m + p_i,m  (for all jobs i)

5. Objective:
   - Minimize C_max (Makespan).

Implementation Note:
-------------------
This solver uses Google OR-Tools CP-SAT, which models disjunctive constraints 
using Interval Variables and the 'AddNoOverlap' constraint.
"""
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
