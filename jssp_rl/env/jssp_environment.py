
import torch
from collections import defaultdict, deque
from config import w1,w2,w3,w4,alpha_idle,delta_bonus


class JSSPEnvironment:
    def __init__(self, times, machines, device=None, use_shaping_rewards=True):
        """
        Args:
            times (Tensor): [num_jobs, num_machines] processing durations
            machines (Tensor): [num_jobs, num_machines] machine IDs
            device (torch.device): Target device (CPU or CUDA)
        """
        self.device = device # or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.times = times.to(dtype=torch.float32, device=self.device)
        self.machines = machines.to(dtype=torch.long, device=self.device)
        self.use_shaping_rewards = use_shaping_rewards


        self.num_jobs = len(times)
        self.num_machines = len(times[0])

        self.reset()

    def reset(self):
        self.state = torch.zeros((self.num_jobs, self.num_machines), dtype=torch.int, device=self.device)
        self.job_completion_times = torch.zeros((self.num_jobs, self.num_machines), device=self.device)
        self.job_start_times = torch.zeros((self.num_jobs, self.num_machines), device=self.device)
        self.machine_available_times = torch.zeros(self.num_machines, device=self.device)


        return self.state
    
    def phi1_scheduled_ratio(self):
        total_ops = self.num_jobs * self.num_machines
        scheduled_ops = self.state.sum().item()
        return scheduled_ops / total_ops

    def phi2_remaining_work(self):
        total_work = float(self.times.sum().item())
        remaining_work = (self.times * (self.state == 0)).sum().item()
        return remaining_work / (total_work + 1e-8)
    
    def phi3_critical_path_length(self):
        """
        Calculates a fast lower bound of the critical path using 
        job-precedence and machine-load bottlenecks instead of full topological sort.
        This is significantly faster for 1000+ operations.
        """
        # Node-level earliest start (predecessor based)
        # We can use the already calculated job_completion_times for scheduled ops
        # and a simple projection for unscheduled ones.
        
        # Max over jobs: (already finished time + remaining work in job)
        job_lb = (self.job_completion_times[:, -1:] + 
                  (self.times * (self.state == 0)).sum(dim=1, keepdim=True)).max().item()
        
        # Max over machines: (current available time + remaining work assigned to machine)
        unscheduled_mask = (self.state == 0).float()
        machine_load = torch.zeros(self.num_machines, device=self.device)
        machine_load.scatter_add_(0, self.machines.flatten(), (self.times * unscheduled_mask).flatten())
        machine_lb = (self.machine_available_times + machine_load).max().item()
        
        critical_path_lb = max(job_lb, machine_lb)
        
        total_work = float(self.times.sum().item())
        return critical_path_lb / (total_work + 1e-8)

    def phi4_max_remaining_ops_per_job(self):
        max_remaining = (self.state == 0).sum(dim=1).max().item()
        return max_remaining / self.num_machines
  
    def penalty_machine_idleness(self):
        horizon = self.job_completion_times.max().item()
        if horizon < 1e-6: return 0.0
        
        # Total work assigned and already scheduled on each machine
        unscheduled_mask = (self.state == 1).float()
        busy_times = torch.zeros(self.num_machines, device=self.device)
        busy_times.scatter_add_(0, self.machines.flatten(), (self.times * unscheduled_mask).flatten())
        
        total_idle = (horizon - busy_times).sum().item()
        return - total_idle / (horizon * self.num_machines + 1e-6)
    
    def bonus_job_completion(self, prev_state):
        # Optimized: check which jobs just became full of 1s
        was_incomplete = (prev_state.sum(dim=1) < self.num_machines)
        is_complete = (self.state.sum(dim=1) == self.num_machines)
        completed_now = (was_incomplete & is_complete).sum().item()
        return float(completed_now)


    def estimate_clb(self):
        """
        Lower bound estimation of remaining makespan (CLB) based on:
        - Max remaining job duration
        - Max machine availability
        """
        job_remaining = []
        for j in range(self.num_jobs):
            rem = 0
            for o in range(self.num_machines):
                if self.state[j, o] == 0:
                    rem += self.times[j, o].item()
            job_remaining.append(rem)

        return max(job_remaining)



    def step(self, action, use_clb_reward=False):
        # --- 0) Bounds guard ---
        if not isinstance(action, int):
            action = int(action)
        if action < 0 or action >= self.num_jobs * self.num_machines:
            print(f"[DEBUG] Invalid action index {action} (out of bounds)")
            return self.state, -200.0, False, float(self.job_completion_times.max().item())

        # --- 1) Decode flat action ---
        job_id = action // self.num_machines
        op_idx = action % self.num_machines

        machine_id = int(self.machines[job_id, op_idx].item())
        duration   = float(self.times[job_id, op_idx].item())

        # --- 2) Invalid move guards ---
        # (a) already scheduled op
        if self.state[job_id, op_idx] == 1:
            print(f"[DEBUG] Invalid: Job {job_id} op {op_idx} already scheduled")
            return self.state, -200.0, False, float(self.job_completion_times.max().item())

        # (b) predecessor not finished yet
        if op_idx > 0 and self.state[job_id, op_idx - 1] == 0:
            print(f"[DEBUG] Invalid: Job {job_id} op {op_idx} predecessor not done")
            return self.state, -200.0, False, float(self.job_completion_times.max().item())
        
        # --- 3) metrics (if needed for rewards) ---
        prev_phi1, prev_phi2, prev_phi3, prev_phi4 = 0, 0, 0, 0
        prev_clb = None
        prev_state = None
        if self.use_shaping_rewards:
            prev_phi1 = self.phi1_scheduled_ratio()
            prev_phi2 = self.phi2_remaining_work()
            prev_phi3 = self.phi3_critical_path_length()
            prev_phi4 = self.phi4_max_remaining_ops_per_job()
            prev_state = self.state.clone()
            if use_clb_reward:
                prev_clb = self.estimate_clb()

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

        # --- 6) New makespan & Reward_base ---
        makespan = float(self.job_completion_times.max().item())
        done = bool(self.state.sum().item() == self.num_jobs * self.num_machines)
        reward = 0.0

        if self.use_shaping_rewards:
            R_base   = -(makespan - prev_makespan)  # less than or equal to 0.0

            # Phi after scheduling
            next_phi1 = self.phi1_scheduled_ratio()
            next_phi2 = self.phi2_remaining_work()
            next_phi3 = self.phi3_critical_path_length()
            next_phi4 = self.phi4_max_remaining_ops_per_job()

            # Shaped Reward
            reward = R_base
            reward += w1 * (next_phi1 - prev_phi1)
            reward += w2 * (next_phi2 - prev_phi2)
            reward += w3 * (next_phi3 - prev_phi3)
            reward += w4 * (next_phi4 - prev_phi4)
            reward += alpha_idle * self.penalty_machine_idleness()
            reward += delta_bonus * self.bonus_job_completion(prev_state)

            # Optional CLB-style reward
            if use_clb_reward:
                next_clb = self.estimate_clb()
                reward = prev_clb - next_clb

        # Keep the signature identical: (state, reward, done, makespan)
        return self.state, float(reward), done, float(makespan)
        
        #print(f"[Reward] R_base={R_base:.2f}, Φ1={prev_phi1:.2f}, Φ2={prev_phi2:.2f}, Φ3={prev_phi3:.2f}, Φ4={prev_phi4:.2f}, idle={self.penalty_machine_idleness():.2f}, bonus={self.bonus_job_completion(prev_state):.2f}, total={reward:.2f}")


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
                    else:
                        
                        print(f"[DEBUG] Job {job}, op {op} blocked "
                            f"(predecessor {op-1} not done)")
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
    
    def get_est_boost(self, valid_mask: torch.Tensor, est_beta, device=None):
        """
        Returns a tensor of the same shape as the logits (one per op),
        with positive boosting for ops with lower estimated start times (EST).

        Args:
            valid_mask (Tensor): Boolean mask of shape [num_total_ops], indicating legal actions.
            device (torch.device): The target device for the returned tensor.
            est_beta (float): Boosting strength (higher → sharper preference for low EST).
        
        Returns:
            Tensor of shape [num_total_ops], where boost[i] > 0 if op i is valid.
        """
        device = device
        num_total_ops = self.num_jobs * self.num_machines

        ests = torch.full((num_total_ops,), float('inf'), device=device)

        for job in range(self.num_jobs):
            for op in range(self.num_machines):
                flat_idx = job * self.num_machines + op

                # Only consider ops allowed by valid_mask
                if not valid_mask[flat_idx]:
                    continue

                # Check if schedulable
                if self.state[job, op] == 1:
                    ests[flat_idx] = 0.0
                    continue

                # Predecessor constraint
                if op > 0 and self.state[job, op - 1] == 0:
                    continue  # not schedulable yet

                # EST = max(predecessor end, machine availability)
                pred_end = self.job_completion_times[job, op - 1] if op > 0 else 0.0
                machine_id = int(self.machines[job, op].item())
                mach_free = self.machine_available_times[machine_id]

                est = max(pred_end, mach_free)
                ests[flat_idx] = est

        # Normalize using softmin: lower EST → higher boost
        ests_valid = ests[valid_mask]
        if ests_valid.numel() == 0:
            return torch.zeros_like(ests)  # no valid ops, return zero boost

        # Softmin normalization: boost = softmin(-β * EST)
        softmin_weights = torch.softmax(-est_beta * ests_valid, dim=0)

        # Map back to full tensor
        boost = torch.zeros_like(ests)
        boost[valid_mask] = softmin_weights

        return boost


