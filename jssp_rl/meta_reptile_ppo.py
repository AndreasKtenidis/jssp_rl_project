
from __future__ import annotations

import copy
import random
from typing import Iterable, Optional

import torch
from torch.optim import Adam
from train.train_ppo import train as ppo_inner_train     
from train.validate_ppo import validate_ppo            
from models.actor_critic_ppo import ActorCriticPPO  
# ————————————————————————————————————————————————————————————————
# Meta‑training routine
# ————————————————————————————————————————————————————————————————

def reptile_meta_train(
    task_loader: Iterable,
    actor_critic: ActorCriticPPO,
    *,
    meta_iterations: int = 1000,
    meta_batch_size: int = 4,
    inner_steps: int = 3,
    inner_lr: float = 3e-4,
    meta_lr: float = 1e-2,
    device: Optional[torch.device] = None,
    val_loader: Optional[Iterable] = None,
) -> ActorCriticPPO:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_critic = actor_critic.to(device)

    # Create a list of *individual tasks* (instances) we can sample from.
    if hasattr(task_loader, "dataset"):
        task_pool = list(task_loader.dataset)
    else:  # Already a list/tuple
        task_pool = list(task_loader)

    assert len(task_pool) >= meta_batch_size, "meta_batch_size exceeds dataset size"

    for meta_iter in range(1, meta_iterations + 1):
        theta_initial = copy.deepcopy(actor_critic.state_dict())
        # Accumulate parameter deltas across tasks
        delta_state = {k: torch.zeros_like(v) for k, v in theta_initial.items()}

        # ——— Sample a meta‑batch of distinct tasks ———
        tasks = random.sample(task_pool, k=meta_batch_size)
        for task in tasks:
            # Build a loader containing just *this* task 
            task_loader_single = [{
                "times": task["times"].unsqueeze(0),
                "machines": task["machines"].unsqueeze(0),
            }]

            # Clone θ to obtain θ′ for the task
            task_model = copy.deepcopy(actor_critic).to(device)
            inner_optim = Adam(task_model.parameters(), lr=inner_lr)

            # Run k inner optimisation steps (PPO epochs) on this task
            for _ in range(inner_steps):
                ppo_inner_train(task_loader_single, task_model, inner_optim, device=device)

            # Accumulate θ′ − θ into delta_state
            adapted_state = task_model.state_dict()
            for name in delta_state:
                delta_state[name] += adapted_state[name] - theta_initial[name]
            del task_model  # free VRAM

        # ——— Reptile outer update: θ ← θ + ε * mean(θ′ − θ) ———
        for name in theta_initial:
            theta_initial[name] += (meta_lr / meta_batch_size) * delta_state[name]
        actor_critic.load_state_dict(theta_initial)

        # ——— Optional logging ———
        if val_loader is not None and meta_iter % 50 == 0:
            actor_critic.eval()
            avg_make = validate_ppo(val_loader, actor_critic, device=device)
            actor_critic.train()
            print(f"[Meta‑Iter {meta_iter:04d}] Validation average makespan: {avg_make:.2f}")

    print(" Meta‑training complete. Use the returned model as a warm‑start θ★.")
    return actor_critic

