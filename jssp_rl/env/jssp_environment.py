
import torch

class JSSPEnvironment:
    def __init__(self, times, machines, device=None):
        """
        Args:
            times (Tensor): [num_jobs, num_machines] processing durations
            machines (Tensor): [num_jobs, num_machines] machine IDs
            device (torch.device): Target device (CPU or CUDA)
        """
        self.device = device # or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.times = times.to(dtype=torch.float32, device=self.device)
        self.machines = machines.to(dtype=torch.long, device=self.device)


        self.num_jobs = len(times)
        self.num_machines = len(times[0])

        self.reset()

    def reset(self):
        self.state = torch.zeros((self.num_jobs, self.num_machines), dtype=torch.int, device=self.device)
        self.job_completion_times = torch.zeros((self.num_jobs, self.num_machines), device=self.device)
        self.job_start_times = torch.zeros((self.num_jobs, self.num_machines), device=self.device)
        self.machine_available_times = torch.zeros(self.num_machines, device=self.device)

        return self.state

    def step(self, action):
        # --- 0) Bounds guard ---
        if not isinstance(action, int):
            action = int(action)
        if action < 0 or action >= self.num_jobs * self.num_machines:
            # invalid flat index
            return self.state, -5.0, False, float(self.job_completion_times.max().item())

        # --- 1) Decode flat action ---
        job_id = action // self.num_machines
        op_idx = action % self.num_machines

        machine_id = int(self.machines[job_id, op_idx].item())
        duration   = float(self.times[job_id, op_idx].item())

        # --- 2) Invalid move guards ---
        # (a) already scheduled op
        if self.state[job_id, op_idx] == 1:
            return self.state, -5.0, False, float(self.job_completion_times.max().item())

        # (b) predecessor not finished yet
        if op_idx > 0 and self.state[job_id, op_idx - 1] == 0:
            return self.state, -5.0, False, float(self.job_completion_times.max().item())

        # --- 3) previous makespan (before scheduling this op) ---
        prev_makespan = float(self.job_completion_times.max().item())

        # --- 4) Feasible start (job precedence & machine availability) ---
        prev_end   = float(self.job_completion_times[job_id, op_idx - 1].item()) if op_idx > 0 else 0.0
        mach_free  = float(self.machine_available_times[machine_id].item() if hasattr(self.machine_available_times, "ndim") else self.machine_available_times[machine_id])
        start_time = max(prev_end, mach_free)
        end_time   = start_time + duration

        # --- 5) Apply schedule ---
        self.job_start_times[job_id, op_idx]      = start_time
        self.job_completion_times[job_id, op_idx] = end_time
        self.machine_available_times[machine_id]  = end_time
        self.state[job_id, op_idx] = 1

        # --- 6) New makespan & reward ---
        makespan = float(self.job_completion_times.max().item())
        reward   = -(makespan - prev_makespan)  # ≤ 0.0

        # --- 7) Done? ---
        done = bool(self.state.sum().item() == self.num_jobs * self.num_machines)

        # Keep the signature identical: (state, reward, done, makespan)
        return self.state, float(reward), done, float(makespan)




    def get_available_actions(self):
        available = []
        for job in range(self.num_jobs):
            for op in range(self.num_machines):
                if self.state[job, op] == 0:
                    if op == 0 or self.state[job, op - 1] == 1:
                        index = job * self.num_machines + op
                        available.append(index)
                    break
        return available

    def get_makespan(self):
        return self.job_completion_times.max().item()

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

