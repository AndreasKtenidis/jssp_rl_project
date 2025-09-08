# meta_reptile_Hetero.py

from __future__ import annotations
import copy
import random
import time
import os
import csv
from typing import Iterable, Optional, List

import torch
from torch.optim import Adam

from train.train_Hetero_ppo import train as ppo_inner_train
from train.validate_Hetero_ppo import validate_ppo
from models.Hetero_actor_critic_ppo import ActorCriticPPO


def reptile_meta_train(
    task_loader: Iterable,
    actor_critic: ActorCriticPPO,
    *,
    meta_iterations: int = 20,
    meta_batch_size: int = 8,
    inner_steps: int = 1,
    inner_lr: float = 3e-4,
    meta_lr: float = 3e-3,
    device: Optional[torch.device] = None,
    val_loader: Optional[Iterable] = None,
    save_path: str = "saved/meta_best.pth",
    # optional knobs to keep inner PPO "light"
    inner_update_batch_size: int = 16,
    validate_every: int = 10,
    # logging
    log_csv_path: str = "/home/aktenidis/JSSPprojects/jssp_rl_project/jssp_rl/outputs/logs/meta_reptile_hetero_log.csv",
) -> ActorCriticPPO:
    """
    Reptile meta-training over PPO with HeteroGIN + CSV logging.

    Logs per meta-iteration:
      - meta_iter, delta_norm, inner_loss_mean, inner_makespan_mean,
        val_makespan (if validated), best_val_makespan, saved_best (0/1), duration_sec
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_critic = actor_critic.to(device)

    # Create task pool from provided loader/iterable
    if hasattr(task_loader, "dataset"):
        task_pool = list(task_loader.dataset)
    else:
        task_pool = list(task_loader)

    assert len(task_pool) >= meta_batch_size, "meta_batch_size exceeds dataset size"

    # --- prepare CSV logging ---
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    if not os.path.exists(log_csv_path):
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "meta_iter",
                "meta_batch_size",
                "inner_steps",
                "inner_lr",
                "meta_lr",
                "delta_norm",
                "inner_loss_mean",
                "inner_makespan_mean",
                "val_makespan",
                "best_val_makespan",
                "saved_best",
                "duration_sec",
            ])

    best_val_makespan = float("inf")

    for meta_iter in range(1, meta_iterations + 1):
        t0 = time.time()
        print(f"\n🔁 Meta-Iteration {meta_iter}/{meta_iterations}")

        # Snapshot current params (list) & init delta accumulators
        initial_params: List[torch.Tensor] = [p.detach().clone() for p in actor_critic.parameters()]
        delta_list: List[torch.Tensor] = [torch.zeros_like(p, device=p.device) for p in actor_critic.parameters()]

        # Sample tasks for this meta-iteration
        tasks = random.sample(task_pool, k=meta_batch_size)

        # For logging inner metrics across tasks
        inner_losses = []
        inner_makespans = []

        for task_idx, task in enumerate(tasks):
            print(f"  ➤ Inner-Task {task_idx+1}/{meta_batch_size}")

            # Lightweight per-task "dataset": just a list with one instance
            task_batch = [task]

            # Fresh copy of the model for the inner adaptation
            task_model = copy.deepcopy(actor_critic).to(device)
            inner_optim = Adam(task_model.parameters(), lr=inner_lr)

            # Inner adaptation
            task_inner_losses = []
            task_inner_makespans = []
            for step in range(inner_steps):
                print(f"    ↪ Inner Step {step+1}/{inner_steps}")
                # Expect ppo_inner_train to return (avg_makespan, epoch_loss)
                avg_mk, ep_loss = ppo_inner_train(
                    task_batch,
                    task_model,
                    inner_optim,
                    device=device,
                    update_batch_size_size=inner_update_batch_size,
                    kl_threshold=0.25
                )
                # Collect per-inner-step metrics
                if ep_loss is not None:
                    task_inner_losses.append(float(ep_loss))
                if avg_mk is not None:
                    task_inner_makespans.append(float(avg_mk))

            # Aggregate per-task
            if task_inner_losses:
                inner_losses.append(sum(task_inner_losses) / len(task_inner_losses))
            if task_inner_makespans:
                inner_makespans.append(sum(task_inner_makespans) / len(task_inner_makespans))

            # Accumulate parameter deltas in-place: Δ += (θ_task − θ_init)
            with torch.no_grad():
                for i, (p_task, p_init) in enumerate(zip(task_model.parameters(), initial_params)):
                    delta_list[i].add_(p_task.data - p_init)

            del task_model  # free ASAP

        # Outer Reptile update: θ ← θ + (meta_lr/meta_batch_size) * Δ
        with torch.no_grad():
            scale = meta_lr / meta_batch_size
            for p, d in zip(actor_critic.parameters(), delta_list):
                p.data.add_(d, alpha=scale)

        # Compute a scalar delta norm for logging (L2 norm of mean delta)
        with torch.no_grad():
            mean_delta_sq = 0.0
            total_elems = 0
            for d in delta_list:
                mean_delta_sq += (d ** 2).sum().item()
                total_elems += d.numel()
            delta_norm = (mean_delta_sq / max(1, total_elems)) ** 0.5

        # Optional validation every N meta-iterations
        saved_best = 0
        val_makespan = None
        if val_loader is not None and (meta_iter % validate_every == 0 or meta_iter == meta_iterations):
            print("\n🔎 Running validation...")
            actor_critic.eval()
            val_makespan = float(validate_ppo(
            val_loader, actor_critic, device=device,
            limit_instances=50,
            best_of_k=10,
            progress_every=10,
            report_greedy=False,   
            return_both=False      
        ))

            actor_critic.train()

            print(f"[Meta-Iter {meta_iter:04d}] Validation Avg Makespan: {val_makespan:.2f}")
            if val_makespan < best_val_makespan:
                best_val_makespan = val_makespan
                torch.save(actor_critic.state_dict(), save_path)
                print(f"✨ New best θ★ saved at {save_path} (makespan: {val_makespan:.2f})")
                saved_best = 1

        # Write CSV row
        duration = time.time() - t0
        row = [
            meta_iter,
            meta_batch_size,
            inner_steps,
            inner_lr,
            meta_lr,
            delta_norm,
            (sum(inner_losses) / len(inner_losses)) if inner_losses else "",
            (sum(inner_makespans) / len(inner_makespans)) if inner_makespans else "",
            val_makespan if val_makespan is not None else "",
            best_val_makespan if best_val_makespan != float("inf") else "",
            saved_best,
            round(duration, 3),
        ]
        with open(log_csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    print("\n✅ Reptile meta-training complete.")
    print(f"Best makespan during meta-validation: {best_val_makespan:.2f}")
    print(f"Final θ★ saved to: {save_path}")
    print(f"Logs saved to: {log_csv_path}")
    return actor_critic
