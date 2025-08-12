from __future__ import annotations

import copy
import random
from typing import Iterable, Optional

import torch
from torch.optim import Adam


from train.train_ppo import train as ppo_inner_train
from train.validate_ppo import validate_ppo
from models.actor_critic_ppo import ActorCriticPPO


def reptile_meta_train(
    task_loader: Iterable,
    actor_critic: ActorCriticPPO,
    *,
    meta_iterations: int = 30,       
    meta_batch_size: int = 8,        
    inner_steps: int = 1,            
    inner_lr: float = 3e-4,
    meta_lr: float = 1e-2,
    device: Optional[torch.device] = None,
    val_loader: Optional[Iterable] = None,
    save_path: str = "saved/meta_best.pth",
    
    inner_update_batch_size_size: int = 4,
    inner_switch_epoch: int = 1,
    validate_every: int = 10,
) -> ActorCriticPPO:
    
    
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_critic = actor_critic.to(device)

    
    if hasattr(task_loader, "dataset"):
        task_pool = list(task_loader.dataset)
    else:
        task_pool = list(task_loader)

    assert len(task_pool) >= meta_batch_size, "meta_batch_size exceeds dataset size"

    best_val_makespan = float("inf")

    for meta_iter in range(1, meta_iterations + 1):
        print(f"\n🔁 Meta-Iteration {meta_iter}/{meta_iterations}")

        
        initial_params = [p.detach().clone() for p in actor_critic.parameters()]
        delta_list = [torch.zeros_like(p, device=p.device) for p in actor_critic.parameters()]

        
        tasks = random.sample(task_pool, k=meta_batch_size)

        for task_idx, task in enumerate(tasks):
            print(f"  ➤ Inner-Task {task_idx+1}/{meta_batch_size}")

           
            task_batch = [task]

            
            task_model = copy.deepcopy(actor_critic).to(device)
            inner_optim = Adam(task_model.parameters(), lr=inner_lr)

            for step in range(inner_steps):
                print(f"    ↪ Inner Step {step+1}/{inner_steps}")
                
                ppo_inner_train(
                    task_batch,
                    task_model,
                    inner_optim,
                    device=device,
                    switch_epoch=inner_switch_epoch,
                    update_batch_size_size=inner_update_batch_size_size,
                )

            # In-place  deltas: Δ += (θ_task − θ_init)
            with torch.no_grad():
                for i, (p_task, p_init) in enumerate(zip(task_model.parameters(), initial_params)):
                    delta_list[i].add_(p_task.data - p_init)

            del task_model

        # Outer update: θ ← θ + (meta_lr/meta_batch_size) * Δ
        with torch.no_grad():
            scale = meta_lr / meta_batch_size
            for p, d in zip(actor_critic.parameters(), delta_list):
                p.data.add_(d, alpha=scale)

        
        if val_loader is not None and (meta_iter % validate_every == 0 or meta_iter == meta_iterations):
            print("\n🔎 Running validation...")
            actor_critic.eval()
            avg_make = validate_ppo(
                val_loader,
                actor_critic,
                device=device,
                limit_instances=50,   # <-- cap validation size
                best_of_k=1,          # set >1 if you want a best-of-K metric
                progress_every=10
            )
            actor_critic.train()

            print(f"[Meta-Iter {meta_iter:04d}] Validation Avg Makespan: {avg_make:.2f}")

            if avg_make < best_val_makespan:
                best_val_makespan = avg_make
                torch.save(actor_critic.state_dict(), save_path)
                print(f"✨ New best θ★ saved at {save_path} (makespan: {avg_make:.2f})")

    print("\n✅ Reptile meta-training complete.")
    print(f"Best makespan during meta-validation: {best_val_makespan:.2f}")
    print(f"Final θ★ saved to: {save_path}")
    return actor_critic
