
import torch

from torch_geometric.data import Batch as HeteroBatch

class RolloutBuffer:
    def __init__(self):
        self.states = []        # List of HeteroData objects
        self.actions = []       # List of action indices
        self.rewards = []       # List of rewards
        self.log_probs = []     # List of log_probs
        self.values = []        # List of state values
        self.dones = []         # List of done flags

        self.returns = []       # GAE returns
        self.advantages = []    # GAE advantages

        self.masks = []         # list[BoolTensor of shape [num_ops] for each state]
        self.topk_masks = []     # top-k mask per state (BoolTensor [num_ops])
        

    def clear(self):
        self.__init__()

    def add(self, state, action, reward, log_prob, value,  done, mask: torch.Tensor, topk_mask: torch.Tensor):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.masks.append(mask.detach().to(dtype=torch.bool, device=mask.device))
        self.topk_masks.append(topk_mask.detach().to(dtype=torch.bool, device=topk_mask.device))
        self.dones.append(done)
        

    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        """
        Standard GAE(λ) on raw rewards/values.
        Produces: self.advantages, self.returns (both lists of tensors).
        """
        advantages = []
        gae = 0.0
        values = self.values + [last_value]

        for step in reversed(range(len(self.rewards))):
            # All are scalar tensors
            r = self.rewards[step]
            v = values[step]
            v_next = values[step + 1]
            done = self.dones[step]

            delta = r + gamma * v_next * (1 - float(done)) - v
            gae = delta + gamma * lam * (1 - float(done)) * gae
            advantages.insert(0, gae)

        self.advantages = advantages
        self.returns = [adv + val for adv, val in zip(self.advantages, self.values)]

    

    def get_batches(self, batch_size: int):
        """
        Yields tuples:
          (
            batch_data: HeteroBatch,
            actions_batch: LongTensor [B],
            old_log_probs_batch: FloatTensor [B],
            returns_batch: FloatTensor [B],
            advantages_batch: FloatTensor [B],
            old_values_batch: FloatTensor [B],
            masks_batch: BoolTensor [sum_ops],
            topk_masks_batch: BoolTensor [sum_ops],
            bias_penalties_batch: FloatTensor [sum_ops],
          )
        """
        total_size = len(self.states)
        if total_size == 0:
            return
        indices = torch.randperm(total_size)

        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]

            # 1) Hetero batch
            data_list = [self.states[i] for i in batch_indices]
            batch_data = HeteroBatch.from_data_list(data_list)

            # 2) Flat tensors (left on CPU; moved to device in train loop)
            actions_batch       = torch.tensor([self.actions[i] for i in batch_indices], dtype=torch.long)
            old_log_probs_batch = torch.stack([self.log_probs[i]   for i in batch_indices], dim=0).view(-1)
            returns_batch       = torch.stack([self.returns[i]     for i in batch_indices], dim=0).view(-1)
            advantages_batch    = torch.stack([self.advantages[i]  for i in batch_indices], dim=0).view(-1)
            old_values_batch    = torch.stack([self.values[i]      for i in batch_indices], dim=0).view(-1)

            # 3) Masks/penalties – concatenated in the same order
            masks_list           = [self.masks[i]           for i in batch_indices]
            topk_masks_list      = [self.topk_masks[i]      for i in batch_indices]
            

            masks_batch          = torch.cat(masks_list, dim=0)
            topk_masks_batch     = torch.cat(topk_masks_list, dim=0)
            

            #  mask/penalty lengths must match #op nodes in the batch
            num_ops_total = int(batch_data['op'].x.size(0))
            if (masks_batch.numel() != num_ops_total or
                topk_masks_batch.numel() != num_ops_total ):
                raise RuntimeError(
                    f"[HeteroBuffer] mismatch: legal={masks_batch.numel()} "
                    f"topk={topk_masks_batch.numel()}  "
                    f"vs #op={num_ops_total}"
                )

            yield (
                batch_data,             # HeteroData 
                actions_batch,          # [B]
                old_log_probs_batch,    # [B]
                returns_batch,          # [B]
                advantages_batch,       # [B]
                old_values_batch,       # [B]
                masks_batch,            # [sum_ops] bool
                topk_masks_batch,       # [sum_ops] bool
                
            )

    def merge(self, other: "RolloutBuffer"):
        # Concatenate lists from another buffer
        self.states          += other.states
        self.actions         += other.actions
        self.rewards         += other.rewards
        self.log_probs       += other.log_probs
        self.values          += other.values
        self.dones           += other.dones

        self.masks           += other.masks
        self.topk_masks      += other.topk_masks
        

        if getattr(other, "returns", None):
            self.returns    += other.returns
        if getattr(other, "advantages", None):
            self.advantages += other.advantages