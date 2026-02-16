
import torch
from collections import defaultdict, deque
from config import w1,w2,w3,w4,alpha_idle,delta_bonus


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
    
    def phi1_scheduled_ratio(self):
        total_ops = self.num_jobs * self.num_machines
        scheduled_ops = self.state.sum().item()
        return scheduled_ops / total_ops

    def phi2_remaining_work(self):
        total_work = float(self.times.sum().item())
        remaining_work = 0.0
        for job in range(self.num_jobs):
            for op in range(self.num_machines):
                if self.state[job, op] == 0:
                    remaining_work += self.times[job, op].item()
        # normalized: 1 = all work left, 0 = no work left
        return remaining_work / total_work  
    
    def phi3_critical_path_length(self):
        

        num_ops = self.num_jobs * self.num_machines
        durations = {}  # (j,o) -> duration
        graph = defaultdict(list)
        in_degree = defaultdict(int)

        # --- Job precedence edges ---
        for j in range(self.num_jobs):
            for o in range(self.num_machines):
                op = (j, o)
                durations[op] = self.times[j, o].item()
                if o < self.num_machines - 1:
                    succ = (j, o + 1)
                    graph[op].append(succ)
                    in_degree[succ] += 1

        # --- Machine constraint edges (only scheduled so far) ---
        machine_ops = defaultdict(list)
        for j in range(self.num_jobs):
            for o in range(self.num_machines):
                if self.state[j, o] == 1:
                    m = int(self.machines[j, o].item())
                    s_time = self.job_start_times[j, o].item()
                    machine_ops[m].append((s_time, (j, o)))

        for m, ops in machine_ops.items():
            ops.sort()
            for i in range(len(ops) - 1):
                _, op_i = ops[i]
                _, op_j = ops[i + 1]
                graph[op_i].append(op_j)
                in_degree[op_j] += 1

        # --- Topological Sort ---
        topo_order = []
        q = deque()
        for op in durations.keys():
            if in_degree[op] == 0:
                q.append(op)

        while q:
            current = q.popleft()
            topo_order.append(current)
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    q.append(neighbor)

        # --- Longest path ---
        longest = {op: 0.0 for op in durations.keys()}
        for op in topo_order:
            for succ in graph[op]:
                cand = longest[op] + durations[op]
                if cand > longest[succ]:
                    longest[succ] = cand

        critical_path = max(longest.values())

        # --- Normalization ---
        total_work = float(self.times.sum().item())
        return critical_path / total_work   # ∈ [0,1]



    def phi4_max_remaining_ops_per_job(self):
        max_remaining = 0
        for job in range(self.num_jobs):
            remaining_ops = 0
            for op in range(self.num_machines):
                if self.state[job, op] == 0:
                    remaining_ops += 1
            if remaining_ops > max_remaining:
                max_remaining = remaining_ops

        # normalization: divide by total ops per job
        return max_remaining / self.num_machines   # ∈ [0,1]
  
    
    def penalty_machine_idleness(self):
        total_idle = 0.0
        horizon = self.job_completion_times.max().item()
        for m in range(self.num_machines):
            busy_time = sum(
                self.times[j, o].item()
                for j in range(self.num_jobs)
                for o in range(self.num_machines)
                if int(self.machines[j, o].item()) == m and self.state[j, o] == 1
            )
            total_idle += horizon - busy_time
        return - total_idle / (horizon * self.num_machines + 1e-6)
  
    
    def bonus_job_completion(self, prev_state):
        completed_now = 0

        for job_id in range(self.num_jobs):
            was_incomplete = (prev_state[job_id] == 0).any()
            is_now_complete = (self.state[job_id] == 1).all()

            if was_incomplete and is_now_complete:
                completed_now += 1

        return +completed_now  


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
        
        # Phi before scheduling
        prev_phi1 = self.phi1_scheduled_ratio()
        prev_phi2 = self.phi2_remaining_work()
        prev_phi3 = self.phi3_critical_path_length()
        prev_phi4 = self.phi4_max_remaining_ops_per_job()
        prev_state = self.state.clone()

        prev_clb = self.estimate_clb() if use_clb_reward else None



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
        R_base   = -(makespan - prev_makespan)  # less than or equal to 0.0

        # Phi after scheduling
        next_phi1 = self.phi1_scheduled_ratio()
        next_phi2 = self.phi2_remaining_work()
        next_phi3 = self.phi3_critical_path_length()
        next_phi4 = self.phi4_max_remaining_ops_per_job()

       # --- 7) Done? ---
        done = bool(self.state.sum().item() == self.num_jobs * self.num_machines)

       # Shaped Reward
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
        
        #print(f"[Reward] R_base={R_base:.2f}, Phi1={prev_phi1:.2f}, Phi2={prev_phi2:.2f}, Phi3={prev_phi3:.2f}, Phi4={prev_phi4:.2f}, idle={self.penalty_machine_idleness():.2f}, bonus={self.bonus_job_completion(prev_state):.2f}, total={reward:.2f}")


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
            est_beta (float): Boosting strength (higher -> sharper preference for low EST).
        
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

        # Softmin normalization: boost = softmin(-beta * EST)
        softmin_weights = torch.softmax(-est_beta * ests_valid, dim=0)

        # Map back to full tensor
        boost = torch.zeros_like(ests)
        boost[valid_mask] = softmin_weights

        return boost


