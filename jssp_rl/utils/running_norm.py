# utils/running_norm.py

from __future__ import annotations
import torch

class RunningReturnNormalizer:
    """
    Tracks running mean/variance of return targets using a numerically stable
    merge (Welford/Chan) so the critic can train on a standardized target.
    """
    def __init__(self, epsilon: float = 1e-8):
        # Running moments
        self.running_mean: float = 0.0
        self.running_var:  float = 1.0
        self.sample_count:  float = epsilon  # small prior to avoid div-by-zero
        self.epsilon:      float = epsilon

    def update(self, values: torch.Tensor) -> None:
        """
        Update running statistics with a new batch of values.
        Expects a 1D tensor; detaches to avoid polluting autograd graph.
        """
        x = values.detach().float().view(-1)
        if x.numel() == 0:
            return

        batch_mean  = x.mean()
        batch_var   = x.var(unbiased=False) + self.epsilon
        batch_count = x.numel()

        # Merge batch stats into running stats (Chan's parallel variance formula)
        delta      = batch_mean - self.running_mean
        total_cnt  = self.sample_count + batch_count

        new_mean = self.running_mean + delta * (batch_count / total_cnt)
        m_a = self.running_var * self.sample_count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + delta * delta * (self.sample_count * batch_count / total_cnt)) / total_cnt

        self.running_mean = float(new_mean)
        self.running_var  = float(new_var)
        self.sample_count = float(total_cnt)

    @property
    def mean(self) -> float:
        return self.running_mean

    @property
    def std(self) -> float:
        # sqrt of variance; add epsilon guard when used
        return float(self.running_var ** 0.5)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Convenience helper: returns (values - mean) / std with minimum std clamp.
        """
        device = values.device
        dtype  = values.dtype
        mean = torch.tensor(self.mean, device=device, dtype=dtype)
        std  = torch.tensor(self.std,  device=device, dtype=dtype).clamp_min(1e-6)
        return (values - mean) / std

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform: undo standardization.
        """
        device = values.device
        dtype  = values.dtype
        mean = torch.tensor(self.mean, device=device, dtype=dtype)
        std  = torch.tensor(self.std,  device=device, dtype=dtype).clamp_min(1e-6)
        return values * std + mean

    # Checkpoint helpers
    def state_dict(self) -> dict:
        return {
            "running_mean": self.running_mean,
            "running_var":  self.running_var,
            "sample_count": self.sample_count,
            "epsilon":      self.epsilon,
        }

    def load_state_dict(self, state: dict) -> None:
        self.running_mean = float(state.get("running_mean", 0.0))
        self.running_var  = float(state.get("running_var", 1.0))
        self.sample_count = float(state.get("sample_count", 1e-8))
        self.epsilon      = float(state.get("epsilon", 1e-8))
