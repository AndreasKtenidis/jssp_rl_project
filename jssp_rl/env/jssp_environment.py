
import torch
from collections import defaultdict, deque
from config import gamma,w1,w2,w3,w4,alpha_idle,delta_bonus

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
        remaining_work = 0.0
        for job in range(self.num_jobs):
            for op in range(self.num_machines):
                if self.state[job, op] == 0:
                    remaining_work += self.times[job, op].item()
        return -remaining_work  
    
    def phi3_critical_path_length(self):
    
        # --- Step 1: Δημιουργία κόμβων ---
        num_ops = self.num_jobs * self.num_machines
        durations = {}  # (j,o) -> duration
        graph = defaultdict(list)   # (j,o) -> list of successors
        in_degree = defaultdict(int)

        # --- Step 2: Job precedence edges ---
        for j in range(self.num_jobs):
            for o in range(self.num_machines):
                op = (j, o)
                durations[op] = self.times[j, o].item()
                if o < self.num_machines - 1:
                    succ = (j, o + 1)
                    graph[op].append(succ)
                    in_degree[succ] += 1

        # --- Step 3: Machine constraint edges ---
        machine_ops = defaultdict(list)  # machine_id -> list of (start_time, (j,o))
        for j in range(self.num_jobs):
            for o in range(self.num_machines):
                if self.state[j, o] == 1:
                    m = int(self.machines[j, o].item())
                    s_time = self.job_start_times[j, o].item()
                    machine_ops[m].append((s_time, (j, o)))

        for m, ops in machine_ops.items():
            ops.sort()  # order by start time
            for i in range(len(ops) - 1):
                _, op_i = ops[i]
                _, op_j = ops[i + 1]
                graph[op_i].append(op_j)
                in_degree[op_j] += 1

        # --- Step 4: Topological Sort ---
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

        # --- Step 5: Longest path via DP ---
        longest = {op: 0.0 for op in durations.keys()}

        for op in topo_order:
            for succ in graph[op]:
                cand = longest[op] + durations[op]
                if cand > longest[succ]:
                    longest[succ] = cand

        # --- Step 6: Return critical path length ---
        return -max(longest.values())  # shaping: return as negative cost


    def phi4_max_remaining_ops_per_job(self):
        max_remaining = 0
        for job in range(self.num_jobs):
            remaining_ops = 0
            for op in range(self.num_machines):
                if self.state[job, op] == 0:
                    remaining_ops += 1
            if remaining_ops > max_remaining:
                max_remaining = remaining_ops
        return -max_remaining  
    
    def penalty_machine_idleness(self):
        idle_machines = 0
        for machine_id in range(self.num_machines):
            # Αν η μηχανή δεν είναι σε χρήση στο παρόν χρονικό σημείο
            last_time = self.machine_available_times[machine_id].item()
            if last_time < self.job_completion_times.max().item():
                idle_machines += 1
        return -idle_machines  
    
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
        
        # Φ before scheduling
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
        R_base   = -(makespan - prev_makespan)  # ≤ 0.0

        # Φ after scheduling
        next_phi1 = self.phi1_scheduled_ratio()
        next_phi2 = self.phi2_remaining_work()
        next_phi3 = self.phi3_critical_path_length()
        next_phi4 = self.phi4_max_remaining_ops_per_job()

       # --- 7) Done? ---
        done = bool(self.state.sum().item() == self.num_jobs * self.num_machines)

       # Shaped Reward
        if done:
            reward = -makespan  # terminal reward
        else:
            reward = R_base #sc5 to simplify reward keep this line in else/delete bellow
            #reward += gamma * (next_phi1 - prev_phi1) * w1
            #reward += gamma * (next_phi2 - prev_phi2) * w2
            #reward += gamma * (next_phi3 - prev_phi3) * w3
            #reward += gamma * (next_phi4 - prev_phi4) * w4
            #reward += alpha_idle    * self.penalty_machine_idleness()
            #reward += delta_bonus   * self.bonus_job_completion(prev_state)

        # Optional CLB-style reward
        if use_clb_reward:
            next_clb = self.estimate_clb()
            reward = prev_clb - next_clb

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

