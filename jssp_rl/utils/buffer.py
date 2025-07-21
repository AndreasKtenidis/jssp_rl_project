import torch

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

        self.returns = []
        self.advantages = []

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

            yield (
                torch.stack([self.states[i] for i in batch_indices]),
                torch.tensor([self.actions[i] for i in batch_indices]),
                torch.tensor([self.log_probs[i] for i in batch_indices]),
                torch.tensor([self.returns[i] for i in batch_indices]),
                torch.tensor([self.advantages[i] for i in batch_indices]),
            )
