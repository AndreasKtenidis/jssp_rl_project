import random

def generate_random_jssp(num_jobs=15, num_machines=15, seed=None):
    if seed is not None:
        random.seed(seed)
    instance = []
    for _ in range(num_jobs):
        machines = random.sample(range(num_machines), num_machines)
        durations = [random.randint(1, 99) for _ in range(num_machines)]
        job = list(zip(machines, durations))
        instance.append(job)
    return instance
