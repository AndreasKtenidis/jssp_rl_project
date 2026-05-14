import torch

try:
    from data.unified.common import Instance, infer_num_machines, validate_instance
except ImportError:
    from jssp_rl.data.unified.common import Instance, infer_num_machines, validate_instance


class FJSSPEnvironment:
    def __init__(self, instance: Instance, device=None, use_shaping_rewards=True):
        """
        Args:
            instance: Unified nested-list instance.
                instance[job_id][operation_index] = [(machine_id, processing_time), ...]
            device (torch.device): Target device (CPU or CUDA)
        """
        self.device = device
        self.instance = instance
        self.use_shaping_rewards = use_shaping_rewards

        validate_instance(self.instance)
        self.num_jobs = len(self.instance)
        self.num_machines = infer_num_machines(self.instance)

        self.op_to_job_op = []
        self.job_op_to_op = {}
        self.action_to_op_machine_duration = []
        self.op_to_actions = {}

        for job_id, job in enumerate(self.instance):
            for op_idx, operation in enumerate(job):
                op_id = len(self.op_to_job_op)
                self.op_to_job_op.append((job_id, op_idx))
                self.job_op_to_op[(job_id, op_idx)] = op_id
                self.op_to_actions[op_id] = []

                for machine_id, duration in operation:
                    action_id = len(self.action_to_op_machine_duration)
                    self.action_to_op_machine_duration.append((op_id, machine_id, duration))
                    self.op_to_actions[op_id].append(action_id)

        self.num_operations = len(self.op_to_job_op)
        self.num_actions = len(self.action_to_op_machine_duration)
        self.max_ops_per_job = max(len(job) for job in self.instance)

        self.reset()

    def reset(self):
        self.state = torch.zeros(self.num_operations, dtype=torch.int, device=self.device)
        self.op_start_times = torch.zeros(self.num_operations, dtype=torch.float32, device=self.device)
        self.op_completion_times = torch.zeros(self.num_operations, dtype=torch.float32, device=self.device)
        self.op_assigned_machines = torch.full(
            (self.num_operations,), -1, dtype=torch.long, device=self.device
        )
        self.machine_available_times = torch.zeros(
            self.num_machines, dtype=torch.float32, device=self.device
        )
        return self.state

    def decode_action(self, action):
        if not isinstance(action, int):
            action = int(action)
        return self.action_to_op_machine_duration[action]

    def get_available_actions(self):
        available = []
        for job_id, job in enumerate(self.instance):
            for op_idx, _ in enumerate(job):
                op_id = self.job_op_to_op[(job_id, op_idx)]
                if self.state[op_id] == 1: # Find first unscheduled operation of this job
                    continue
                if op_idx == 0: # If it is the first operation, it is available
                    available.extend(self.op_to_actions[op_id])
                else:
                    prev_op_id = self.job_op_to_op[(job_id, op_idx - 1)]
                    if self.state[prev_op_id] == 1:
                        available.extend(self.op_to_actions[op_id])
                break
        return available
    
    def get_action_mask(self):
        mask = torch.zeros(self.num_actions, dtype=torch.bool, device=self.device)
        available_actions = self.get_available_actions()
        if available_actions:
            mask[available_actions] = True
        return mask

    def step(self, action):
        if not isinstance(action, int):
            action = int(action)
        if action < 0 or action >= self.num_actions:
            print(f"[DEBUG] Invalid action index {action} (out of bounds)")
            return self.state, -200.0, False, self.get_makespan()
        if action not in self.get_available_actions():
            print(f"[DEBUG] Invalid: action {action} is not currently available")
            return self.state, -200.0, False, self.get_makespan()

        op_id, machine_id, duration = self.decode_action(action)
        job_id, op_idx = self.op_to_job_op[op_id]

        if self.state[op_id] == 1:
            print(f"[DEBUG] Invalid: Job {job_id} op {op_idx} already scheduled")
            return self.state, -200.0, False, self.get_makespan()

        if op_idx > 0:
            prev_op_id = self.job_op_to_op[(job_id, op_idx - 1)]
            if self.state[prev_op_id] == 0:
                print(f"[DEBUG] Invalid: Job {job_id} op {op_idx} predecessor not done")
                return self.state, -200.0, False, self.get_makespan()
            prev_end = float(self.op_completion_times[prev_op_id].item())
        else:
            prev_end = 0.0

        prev_makespan = self.get_makespan()
        mach_free = float(self.machine_available_times[machine_id].item())
        start_time = max(prev_end, mach_free)
        end_time = start_time + float(duration)

        self.op_start_times[op_id] = start_time
        self.op_completion_times[op_id] = end_time
        self.op_assigned_machines[op_id] = machine_id
        self.machine_available_times[machine_id] = end_time
        self.state[op_id] = 1

        makespan = self.get_makespan()
        done = bool(self.state.sum().item() == self.num_operations)
        reward = -makespan if done else -(makespan - prev_makespan)

        return self.state, float(reward), done, float(makespan)
    
    def get_makespan(self):
        return float(self.op_completion_times.max().item())

    def extract_job_assignments(self):
        assignments = []
        for op_id, (job_id, op_idx) in enumerate(self.op_to_job_op):
            if self.state[op_id] == 1:
                start = float(self.op_start_times[op_id].item())
                end = float(self.op_completion_times[op_id].item())
                machine = int(self.op_assigned_machines[op_id].item())
                assignments.append({
                    "job_id": job_id,
                    "operation_index": op_idx,
                    "machine": machine,
                    "start_time": start,
                    "end_time": end,
                })
        return assignments
