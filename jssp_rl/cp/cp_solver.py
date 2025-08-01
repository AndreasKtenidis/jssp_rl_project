# jssp_rl/cp/cp_solver.py

from ortools.sat.python import cp_model
from utils.logging_utils import plot_gantt_chart
import os

def solve_instance_your_version(times, machines):
    num_jobs, num_machines = times.shape
    model = cp_model.CpModel()

    jobs_starts, jobs_ends, intervals = {}, {}, {}

    horizon = int(times.sum().item())
    for job in range(num_jobs):
        for op in range(num_machines):
            dur = int(times[job][op])
            start = model.NewIntVar(0, horizon, f'start_{job}_{op}')
            end = model.NewIntVar(0, horizon, f'end_{job}_{op}')
            interval = model.NewIntervalVar(start, dur, end, f'interval_{job}_{op}')
            jobs_starts[(job, op)] = start
            jobs_ends[(job, op)] = end
            intervals[(job, op)] = interval

    # Precedence
    for job in range(num_jobs):
        for op in range(1, num_machines):
            model.Add(jobs_starts[(job, op)] >= jobs_ends[(job, op - 1)])

    # Machine no-overlap
    machine_intervals = {}
    for job in range(num_jobs):
        for op in range(num_machines):
            machine = int(machines[job][op])
            machine_intervals.setdefault(machine, []).append(intervals[(job, op)])
    for m in machine_intervals:
        model.AddNoOverlap(machine_intervals[m])

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, [jobs_ends[(j, num_machines - 1)] for j in range(num_jobs)])
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = []
        for job in range(num_jobs):
            for op in range(num_machines):
                start = solver.Value(jobs_starts[(job, op)])
                end = solver.Value(jobs_ends[(job, op)])
                machine = int(machines[job][op])
                schedule.append((job, machine, start, end))
        return solver.ObjectiveValue(), schedule  

    return None, []  


def run_cp_on_all(instances, save_gantt_dir=None):
    results = []

    for i, inst in enumerate(instances):
        print(f"🧠 Solving CP on instance {i+1}/{len(instances)}")
        
        ms, schedule = solve_instance_your_version(inst["times"], inst["machines"])
        results.append({"instance_id": i, "cp_makespan": ms})

        if schedule and save_gantt_dir:
            save_path = os.path.join(save_gantt_dir, f"gantt_cp_taillardV2_{i:02d}.png")
            converted = [
                {
                    'job_id': job,
                    'machine': machine,
                    'start_time': start,
                    'end_time': end
                }
                for (job, machine, start, end) in schedule
            ]

            plot_gantt_chart(converted, save_path=save_path,show_op_index=True)
            

    return results


