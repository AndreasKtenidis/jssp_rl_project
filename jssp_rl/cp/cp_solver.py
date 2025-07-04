# jssp_rl/cp/cp_solver.py

from ortools.sat.python import cp_model

def solve_instance_with_cp(instance):
    times = instance["times"]
    machines = instance["machines"]
    num_jobs, num_tasks = times.shape
    model = cp_model.CpModel()

    all_tasks = {}
    all_machines = {}

    horizon = int(times.sum().item())  # upper bound on makespan

    for job in range(num_jobs):
        for task in range(num_tasks):
            machine = int(machines[job][task])
            duration = int(times[job][task])
            suffix = f"_{job}_{task}"
            start = model.NewIntVar(0, horizon, "start" + suffix)
            end = model.NewIntVar(0, horizon, "end" + suffix)
            interval = model.NewIntervalVar(start, duration, end, "interval" + suffix)

            all_tasks[(job, task)] = (start, end, interval)
            all_machines.setdefault(machine, []).append(interval)

    # Machine constraints: no overlap
    for machine in all_machines:
        model.AddNoOverlap(all_machines[machine])

    # Job precedence constraints
    for job in range(num_jobs):
        for task in range(num_tasks - 1):
            model.Add(all_tasks[(job, task + 1)][0] >= all_tasks[(job, task)][1])

    # Makespan objective
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(
        makespan,
        [all_tasks[(job, num_tasks - 1)][1] for job in range(num_jobs)]
    )
    model.Minimize(makespan)

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver.ObjectiveValue()
    else:
        return None

def run_cp_on_all(instances, limit=None):
    from tqdm import tqdm
    results = []

    if limit:
        instances = instances[:limit]

    for i, inst in enumerate(tqdm(instances, desc="Solving with CP")):
        try:
            result = solve_instance_with_cp(inst)
            results.append(result)
        except Exception as e:
            print(f"⚠️ CP failed on instance {i}: {e}")
            results.append(None)
    return results
