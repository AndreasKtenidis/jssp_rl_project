# jssp_rl/cp/cp_solver.py
"""
Job Shop Scheduling Problem (JSSP) - CP-SAT Formulation
"""
from ortools.sat.python import cp_model
from utils.logging_utils import plot_gantt_chart
import os
import time
import numpy as np
import torch

def _to_py_int(x):
    # Safely convert torch/numpy scalars to Python int
    if isinstance(x, torch.Tensor):
        return int(x.item())
    if isinstance(x, (np.integer, np.floating)):
        return int(x)
    return int(x)

class JSSPSolutionCallback(cp_model.CpSolverSolutionCallback):
    """Callback to record time of best solution and stop if BKS reached."""
    def __init__(self, makespan_var, bks=None):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._makespan_var = makespan_var
        self._bks = bks
        self.best_makespan = float('inf')
        self.time_to_best = 0.0
        self.start_time = time.time()
        self.status = "FEASIBLE"

    def on_solution_callback(self):
        current_makespan = self.Value(self._makespan_var)
        current_time = time.time() - self.start_time
        
        if current_makespan < self.best_makespan:
            self.best_makespan = current_makespan
            self.time_to_best = current_time
            
            if self._bks is not None:
                if self.best_makespan < self._bks:
                    self.status = "BKS_BEATEN"
                    self.StopSearch()
                elif self.best_makespan == self._bks:
                    self.status = "BKS_REACHED"
                    self.StopSearch()

def find_critical_path(schedule):
    """
    Identifies operations on the critical path of a given schedule.
     schedule: List[dict] with job_id, operation_index, machine, start_time, end_time
    Returns: Set of (job_id, op_index)
    """
    if not schedule: return set()
    
    makespan = max(entry['end_time'] for entry in schedule)
    lookup = {(entry['job_id'], entry['operation_index']): entry for entry in schedule}
    
    # Machine sequences and index lookup for O(1) predecessor access
    machine_map = {}
    for entry in schedule:
        machine_map.setdefault(entry['machine'], []).append(entry)
    
    op_to_m_idx = {}
    for m in machine_map:
        machine_map[m] = sorted(machine_map[m], key=lambda x: x['start_time'])
        for idx, entry in enumerate(machine_map[m]):
            op_to_m_idx[(entry['job_id'], entry['operation_index'])] = idx

    cp_ops = set()
    visited = set()
    stack = []

    # Start from any op that ends at makespan
    for entry in schedule:
        if entry['end_time'] == makespan:
            stack.append(entry)

    while stack:
        curr = stack.pop()
        key = (curr['job_id'], curr['operation_index'])
        if key in visited: continue
        visited.add(key)
        cp_ops.add(key)
        
        s = curr['start_time']
        if s == 0: continue
        
        # 1. Job precedence check
        if curr['operation_index'] > 0:
            prev_j = lookup.get((curr['job_id'], curr['operation_index'] - 1))
            if prev_j and prev_j['end_time'] == s:
                stack.append(prev_j)
        
        # 2. Machine precedence check
        idx = op_to_m_idx.get(key)
        if idx is not None and idx > 0:
            prev_m = machine_map[curr['machine']][idx - 1]
            if prev_m['end_time'] == s:
                stack.append(prev_m)

    return cp_ops

def solve_instance_your_version(times, machines, time_limit_s: float = 10.0, **kwargs):
    """
    Solve one JSSP instance with CP-SAT.
    Args:
        times: [J, M] torch.Tensor or np.ndarray or list of lists
        machines: [J, M] same as above; machine IDs must be 0..M-1
        time_limit_s: CP-SAT wall time in seconds
        kwargs: includes 'bks'
    Returns:
        (makespan:int, schedule:list[dict], status:str, time_to_best:float)
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
    solver.parameters.num_search_workers = 8 

    # We call with bks if provided
    bks = kwargs.get('bks', None)
    cb = JSSPSolutionCallback(makespan, bks=bks)
    status = solver.Solve(model, cb)

    final_status = "FEASIBLE"
    if status == cp_model.OPTIMAL:
        final_status = "OPTIMAL"
    elif cb.status == "BKS_REACHED":
        final_status = "BKS_REACHED"
    elif cb.status == "BKS_BEATEN":
        final_status = "BKS_BEATEN"
    elif status == cp_model.INFEASIBLE:
        final_status = "INFEASIBLE"
    elif status == cp_model.MODEL_INVALID:
        final_status = "INVALID"

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
        return solver.ObjectiveValue(), schedule, final_status, cb.time_to_best

    return None, [], "FAILED", 0.0

def solve_with_rl_warmstart(times, machines, rl_schedule, time_limit_s: float = 60.0, fix_ratio: float = 0.0, **kwargs):
    """
    Solve JSSP using RL schedule as a warm start.
    Args:
        rl_schedule: List[dict] from env.extract_job_assignments()
        fix_ratio: 0.0 to 1.0. If > 0, 'hard fixes' this ratio of operations to RL start times.
        kwargs: includes 'bks'
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
        rl_schedule_sorted = sorted(rl_schedule, key=lambda x: x['start_time'])
        num_to_fix = int(len(rl_schedule_sorted) * fix_ratio)
        
        for i, entry in enumerate(rl_schedule_sorted):
            j, o = entry['job_id'], entry['operation_index']
            rl_start = int(entry['start_time'])
            rl_end = int(entry['end_time'])
            
            if i < num_to_fix:
                model.Add(jobs_starts[(j, o)] == rl_start)
            else:
                model.AddHint(jobs_starts[(j, o)], rl_start)
            
            rl_makespan = max(rl_makespan, rl_end)

    # 4. Objective with RL Upper Bound
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, [jobs_ends[(j, num_machines - 1)] for j in range(num_jobs)])
    
    if rl_makespan > 0:
        model.Add(makespan <= int(rl_makespan))
    
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8 
    
    # Optional BKS stopping
    bks = kwargs.get('bks', None)
    
    # RL vs BKS check
    if bks is not None:
        if rl_makespan < bks:
            rl_status = "BKS_BEATEN_BY_RL"
        elif rl_makespan <= bks:
            rl_status = "BKS_REACHED_BY_RL"
        else:
            rl_status = None
            
        if rl_status:
            # Construct and return immediately
            final_schedule = []
            for j in range(num_jobs):
                for o in range(num_machines):
                    final_schedule.append({
                        "job_id": j,
                        "operation_index": o,
                        "machine": _to_py_int(machines_np[j, o]),
                        "start_time": next(e['start_time'] for e in rl_schedule if e['job_id']==j and e['operation_index']==o),
                        "end_time": next(e['end_time'] for e in rl_schedule if e['job_id']==j and e['operation_index']==o)
                    })
            return rl_makespan, final_schedule, rl_status, 0.1

    cb = JSSPSolutionCallback(makespan, bks=bks)
    status = solver.Solve(model, cb)

    final_status = "FEASIBLE"
    if status == cp_model.OPTIMAL:
        final_status = "OPTIMAL"
    elif cb.status == "BKS_REACHED":
        final_status = "BKS_REACHED"
    elif cb.status == "BKS_BEATEN":
        final_status = "BKS_BEATEN"

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
        return solver.ObjectiveValue(), final_schedule, final_status, cb.time_to_best

    return None, [], "FAILED", 0.0

def solve_hybrid_advanced(times, machines, rl_schedule, mode='machine_precedence', fix_ratio=0.0, **kwargs):
    """
    Advanced Hybrid Solver:
    - 'machine_precedence': Fixes machine order for CP operations.
    - 'cp_fixation': Hard fixes a ratio of CP operations.
    - 'cp_hint': Hints only CP operations.
    """
    if isinstance(times, torch.Tensor): times_np = times.detach().cpu().numpy()
    else: times_np = np.asarray(times)
    if isinstance(machines, torch.Tensor): machines_np = machines.detach().cpu().numpy()
    else: machines_np = np.asarray(machines)

    num_jobs, num_machines = times_np.shape
    model = cp_model.CpModel()
    jobs_starts, jobs_ends, intervals = {}, {}, {}
    horizon = int(times_np.sum())

    for j in range(num_jobs):
        for o in range(num_machines):
            dur = _to_py_int(times_np[j, o])
            s = model.NewIntVar(0, horizon, f's_{j}_{o}')
            e = model.NewIntVar(0, horizon, f'e_{j}_{o}')
            i = model.NewIntervalVar(s, dur, e, f'i_{j}_{o}')
            jobs_starts[(j,o)], jobs_ends[(j,o)], intervals[(j,o)] = s, e, i

    for j in range(num_jobs):
        for o in range(1, num_machines):
            model.Add(jobs_starts[(j, o)] >= jobs_ends[(j, o-1)])
            
    machine_intervals = {}
    for j in range(num_jobs):
        for o in range(num_machines):
            m = _to_py_int(machines_np[j, o])
            machine_intervals.setdefault(m, []).append((j, o))
    
    for m, ops in machine_intervals.items():
        model.AddNoOverlap([intervals[op] for op in ops])

    # --- ADVANCED LOGIC ---
    cp_ops = find_critical_path(rl_schedule)
    rl_lookup = {(e['job_id'], e['operation_index']): e for e in rl_schedule}
    
    if mode == 'machine_precedence':
        # Fix machine sequence for CP ops
        for m, m_ops_keys in machine_intervals.items():
            # Filter RL schedule for this machine and sort by start time
            m_sched = sorted([rl_lookup[k] for k in m_ops_keys], key=lambda x: x['start_time'])
            for i in range(len(m_sched)-1):
                op1 = (m_sched[i]['job_id'], m_sched[i]['operation_index'])
                op2 = (m_sched[i+1]['job_id'], m_sched[i+1]['operation_index'])
                # If both are in CP, or even if one is? Let's fix ALL machine sequences derived from RL
                # as requested ("In machine where CP passes, keep same order")
                if op1 in cp_ops or op2 in cp_ops:
                    model.Add(jobs_starts[op2] >= jobs_ends[op1])
        # Also give hints for everything
        for entry in rl_schedule:
            model.AddHint(jobs_starts[(entry['job_id'], entry['operation_index'])], int(entry['start_time']))

    elif mode == 'cp_fixation':
        cp_list = sorted(list(cp_ops), key=lambda x: rl_lookup[x]['start_time'])
        num_to_fix = int(len(cp_list) * fix_ratio)
        for i, op_key in enumerate(cp_list):
            if i < num_to_fix:
                model.Add(jobs_starts[op_key] == int(rl_lookup[op_key]['start_time']))
            else:
                model.AddHint(jobs_starts[op_key], int(rl_lookup[op_key]['start_time']))
        # Hints for non-CP ops
        for k in rl_lookup:
            if k not in cp_ops:
                model.AddHint(jobs_starts[k], int(rl_lookup[k]['start_time']))

    elif mode == 'cp_hint':
        for k, entry in rl_lookup.items():
            if k in cp_ops:
                # Strong Hint (actually OR-Tools doesn't have weight, so we just hint these)
                model.AddHint(jobs_starts[k], int(entry['start_time']))
            # We skip hints for non-CP or provide them as secondary? 
            # User said "while in others not", so we only hint CP.

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, [jobs_ends[(j, num_machines - 1)] for j in range(num_jobs)])
    
    rl_makespan = max(e['end_time'] for e in rl_schedule)
    model.Add(makespan <= int(rl_makespan))
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(kwargs.get('time_limit_s', 60.0))
    solver.parameters.num_search_workers = 8 
    bks = kwargs.get('bks', None)
    cb = JSSPSolutionCallback(makespan, bks=bks)
    status_code = solver.Solve(model, cb)

    status = "FEASIBLE"
    if status_code == cp_model.OPTIMAL: status = "OPTIMAL"
    elif cb.status == "BKS_REACHED": status = "BKS_REACHED"
    elif cb.status == "BKS_BEATEN": status = "BKS_BEATEN"

    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sched = []
        for j in range(num_jobs):
            for o in range(num_machines):
                sched.append({
                    "job_id": j, "operation_index": o, 
                    "machine": _to_py_int(machines_np[j, o]),
                    "start_time": solver.Value(jobs_starts[(j, o)]),
                    "end_time": solver.Value(jobs_ends[(j, o)])
                })
        return solver.ObjectiveValue(), sched, status, cb.time_to_best
    return None, [], "FAILED", 0.0

def run_cp_on_all(instances, save_gantt_dir=None, time_limit_s: float = 10.0):
    results = []
    for i, inst in enumerate(instances):
        print(f"🧠 Solving CP on instance {i+1}/{len(instances)}")
        ms, _, status, t_best = solve_instance_your_version(inst["times"], inst["machines"], time_limit_s=time_limit_s)
        results.append({"instance_id": i, "cp_makespan": ms, "status": status, "time_to_best": t_best})
    return results
