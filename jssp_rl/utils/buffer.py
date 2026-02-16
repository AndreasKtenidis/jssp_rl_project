import torch
from torch_geometric.data import Batch

class RolloutBuffer:
    def __init__(self):
        self.states = []       # list[Data]
        self.actions = []      # list[int] (scalar actions)
        self.rewards = []      # list[float]
        self.log_probs = []    # list[tensor scalar]
        self.values = []       # list[tensor scalar]
        self.dones = []        # list[bool]
        self.masks = []        # list[BoolTensor shape [num_ops]]

        self.returns = []      # list[tensor scalar]
        self.advantages = []   # list[tensor scalar]


    def clear(self):
        self.__init__()

    def add(self, state, action, reward, log_prob, value, done, mask: torch.Tensor):
        self.states.append(state)
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.dones.append(bool(done))
        self.masks.append(mask.detach().to(dtype=torch.bool, device=mask.device))

    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0
        values = self.values + [last_value]

        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)

        self.advantages = advantages
        self.returns = [adv + val for adv, val in zip(self.advantages, self.values)]

    def get_batches(self, batch_size: int):
        total_size = len(self.states)
        if total_size == 0:
            return
        indices = torch.randperm(total_size)

        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]

            
            data_list = [self.states[i] for i in batch_indices]
            batch_data: Batch = Batch.from_data_list(data_list)

            # 2) flat tensors ( 1-D)
            actions_batch       = torch.tensor([self.actions[i]    for i in batch_indices], dtype=torch.long)
            old_log_probs_batch = torch.stack([self.log_probs[i]   for i in batch_indices], dim=0).view(-1)
            returns_batch       = torch.stack([self.returns[i]     for i in batch_indices], dim=0).view(-1)
            advantages_batch    = torch.stack([self.advantages[i]  for i in batch_indices], dim=0).view(-1)
            old_values_batch    = torch.stack([self.values[i]      for i in batch_indices], dim=0).view(-1)

            # 3) concatenate masks 
            masks_list  = [self.masks[i] for i in batch_indices]
            masks_batch = torch.cat(masks_list, dim=0)  # BoolTensor

            # sanity check: len logits == len masks
            if hasattr(batch_data, 'x') and batch_data.x is not None:
                num_nodes_total = int(batch_data.x.size(0))
            else:
                num_nodes_total = int(batch_data['op'].x.size(0))
            if masks_batch.numel() != num_nodes_total:
                raise RuntimeError(
                    f"[GNN Buffer] masks_batch length ({masks_batch.numel()}) "
                    f"!= num_nodes in batch ({num_nodes_total}). "
                    "Be sure your mask has the same length with #ops of state."
                )

            yield (
                batch_data,
                actions_batch,
                old_log_probs_batch,
                returns_batch,
                advantages_batch,
                old_values_batch,
                masks_batch
            )

    def merge(self, other: "RolloutBuffer"):
        
        
        self.states     += other.states
        self.actions    += other.actions
        self.rewards    += other.rewards
        self.log_probs  += other.log_probs
        self.values     += other.values
        self.dones      += other.dones
        self.masks      += other.masks

        
        if getattr(other, "returns", None):
            self.returns    += other.returns
        if getattr(other, "advantages", None):
            self.advantages += other.advantages
