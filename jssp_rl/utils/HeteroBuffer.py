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

    def clear(self):
        self.__init__()

    def add(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

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

    def get_batches(self, batch_size):
        total_size = len(self.states)
        indices = torch.randperm(total_size)

        for start in range(0, total_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            data_list = [self.states[i] for i in batch_indices]
            batch_data = HeteroBatch.from_data_list(data_list)  

            actions_batch = torch.tensor([self.actions[i] for i in batch_indices], dtype=torch.long)
            log_probs_batch = torch.stack([self.log_probs[i] for i in batch_indices])
            returns_batch = torch.stack([self.returns[i] for i in batch_indices])
            advantages_batch = torch.stack([self.advantages[i] for i in batch_indices])

            yield batch_data, actions_batch, log_probs_batch, returns_batch, advantages_batch

    def merge(self, other):
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.log_probs.extend(other.log_probs)
        self.values.extend(other.values)
        self.dones.extend(other.dones)
        self.returns.extend(other.returns)
        self.advantages.extend(other.advantages)
