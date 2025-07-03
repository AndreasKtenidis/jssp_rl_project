

import torch

class JSSPEnvironment:
    def __init__(self, times, machines):
        """
        Args:
            times (Tensor): [num_jobs, num_machines] processing durations
            machines (Tensor): [num_jobs, num_machines] machine IDs
        """
        self.times = times
        self.machines = machines

        self.num_jobs = times.size(0)
        self.num_machines = times.size(1)

        self.reset()

    def reset(self):
        self.state = torch.zeros((self.num_jobs, self.num_machines), dtype=torch.int)
        self.job_completion_times = torch.zeros((self.num_jobs, self.num_machines))
        self.job_start_times = torch.zeros((self.num_jobs, self.num_machines))
        self.machine_available_times = torch.zeros(self.num_machines)

        return self.state

    def step(self, action):
        job_id = action // self.num_machines
        op_idx = action % self.num_machines
        machine_id = self.machines[job_id, op_idx].item()
        duration = self.times[job_id, op_idx].item()

        # Check precedence
        if op_idx > 0 and self.state[job_id, op_idx - 1] == 0:
            return self.state, -1e6, False, 0  # Invalid move

        prev_end = self.job_completion_times[job_id, op_idx - 1] if op_idx > 0 else 0
        machine_free = self.machine_available_times[machine_id]

        start_time = max(prev_end, machine_free)
        end_time = start_time + duration

        self.job_start_times[job_id, op_idx] = start_time
        self.job_completion_times[job_id, op_idx] = end_time
        self.machine_available_times[machine_id] = end_time
        self.state[job_id, op_idx] = 1

        makespan = self.job_completion_times.max().item()
        reward = -makespan  # Goal: minimize makespan

        done = self.state.sum().item() == self.num_jobs * self.num_machines

        return self.state, reward, done, makespan
    
    def extract_job_assignments(self):
        assignments = []
        for job in range(self.num_jobs):
            for op in range(self.num_machines):
                if self.state[job, op] == 1:
                    start = self.job_start_times[job, op].item()
                    end = self.job_completion_times[job, op].item()
                    machine = self.machines[job, op].item()
                    assignments.append({
                        "job_id": job,
                        "operation_index": op,
                        "machine": machine,
                        "start_time": start,
                        "end_time": end
                    })
        return assignments

