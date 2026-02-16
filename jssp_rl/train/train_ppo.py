import gc
import os
import torch
import torch.nn.functional as F
from torch.distributions import Categorical  # kept for warmup log_prob
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data

from utils.buffer import RolloutBuffer
from utils.features import make_hetero_data
from utils.action_masking import select_top_k_actions
# from models.gnn import GNNWithAttention  <-- removed
from env.jssp_environment import JSSPEnvironment
from utils.train_diagnostics import log_train_metrics_to_csv
from utils.policy_utils import per_graph_logprob_and_entropy  # per-graph logprob/entropy

from config import (
    batch_size, epochs, gamma, gae_lambda,
    clip_epsilon, value_coef, entropy_coef, lr
)

def train(
    dataloader,
    actor_critic,
    optimizer,
    device,
    switch_epoch: int = 3,
    update_batch_size_size: int = 32,
    value_clip_range: float = 0.2,   # PPO2-style value clipping
    kl_threshold: float = 0.02,      # early-stop threshold per chunk
    log_every: int = 20,
    log_dir=None
):
    # default logs dir
    if log_dir is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))  # .../jssp_rl
        log_dir = os.path.join(base_dir, "outputs", "logs")

    actor_critic.train()
    all_makespans = []
    total_episodes = 0

    print("\n=== PPO (GNN) Training in update-batches ===")

    dataset = dataloader.dataset if hasattr(dataloader, "dataset") else dataloader

    for chunk_start in range(0, len(dataset), update_batch_size_size):
        # ---- select rollout chunk ----
        chunk = dataset[chunk_start:chunk_start + update_batch_size_size]
        chunk_buffer = RolloutBuffer()

        # ---- Cosine LR per chunk ----
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

        # =========================
        # [1] Collect Rollouts
        # =========================
        for instance in chunk:
            times    = instance['times']
            machines = instance['machines']

            env = JSSPEnvironment(times, machines, device=device)  # keep device consistent
            env.reset()

            done = False
            ep_reward = 0.0
            ep_buffer = RolloutBuffer()

            while not done:
                data = make_hetero_data(env, device)
                with torch.no_grad():
                    logits, value = actor_critic(data)  # logits: [num_ops]

                if torch.isnan(logits).any():
                    raise ValueError("NaNs in logits - check inputs/network.")

                # available actions & mask
                avail = env.get_available_actions()
                mask = torch.zeros(data['op'].x.size(0), dtype=torch.bool, device=device)
                if len(avail) > 0:
                    mask[avail] = True
                # always mask logits (even during warmup) for consistency
                masked_logits = logits.masked_fill(~mask, -1e10)

                if total_episodes < switch_epoch:
                    # small heuristic warmup
                    _, _, topk_actions, _ = select_top_k_actions(env, top_k=5, device=device)
                    if topk_actions.numel() == 0:
                        # fallback: if no heuristic found, use avail
                        if len(avail) == 0:
                            # should not happen in normal flow, so bail
                            break
                        action = int(avail[0])
                    else:
                        if torch.rand(1).item() < 0.1:
                            action = int(topk_actions[torch.randint(len(topk_actions), (1,))])
                        else:
                            action = int(topk_actions[0])
                    # log_prob from masked logits
                    log_prob = torch.log_softmax(masked_logits, dim=0)[action]
                else:
                    dist = Categorical(logits=masked_logits)
                    action = int(dist.sample().item())
                    log_prob = dist.log_prob(torch.tensor(action, device=device))

                _, reward, done, _ = env.step(action)
                ep_reward += reward
                # save also the mask for this state
                ep_buffer.add(data, action, reward, log_prob, value, done, mask=mask)

            # bootstrap value for last state
            with torch.no_grad():
                final_data = make_hetero_data(env, device)
                _, final_value = actor_critic(final_data)

            ep_buffer.compute_returns_and_advantages(final_value.squeeze(), gamma, gae_lambda)
            chunk_buffer.merge(ep_buffer)

            mk = env.get_makespan()
            all_makespans.append(mk)
            total_episodes += 1
            print(f"[chunk {chunk_start//update_batch_size_size + 1} | Ep {total_episodes}] "
                  f"Makespan: {mk:.2f} | Total Reward: {ep_reward:.2f}")

        print(f"\n=== PPO Update for chunk starting at {chunk_start} ===")

        # =========================
        # [2] PPO UPDATES (PPO2-style)
        # =========================
        for epoch in range(epochs):
            epoch_loss_sum = 0.0
            epoch_batches  = 0
            # entropy decay across epochs
            ent_coef = entropy_coef * (1.0 - epoch / max(1, epochs))

            for b_idx, batch in enumerate(chunk_buffer.get_batches(batch_size)):
                # buffer must return also masks_batch (last element)
                (states, actions, old_log_probs, returns, advantages,
                 old_values, masks_batch) = batch

                # to device
                states        = states.to(device)
                actions       = actions.to(device)
                old_log_probs = old_log_probs.to(device)
                returns       = returns.to(device)
                advantages    = advantages.to(device)
                old_values    = old_values.to(device)
                masks_batch   = masks_batch.to(device)

                # normalize targets
                if returns.numel() > 1:
                    returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
                if advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

                # Forward
                logits, values = actor_critic(states)   # logits: [sum_nodes_in_batch]
                # mask also in updates (same as rollout)
                logits = logits.masked_fill(~masks_batch, -1e10)

                # flatten targets
                values     = values.view(-1)
                returns    = returns.view(-1).detach()
                old_values = old_values.view(-1).detach()

                if torch.isnan(logits).any():
                    raise ValueError("NaNs in logits during update.")

                # per-graph logprobs/entropy (not a huge Categorical across nodes)
                if hasattr(states, 'batch') and states.batch is not None:
                     batch_vec = states.batch
                else:
                     batch_vec = states['op'].batch  # HeteroData batch
                new_log_probs, entropy = per_graph_logprob_and_entropy(
                    logits.view(-1), batch_vec, actions
                )

                # KL approx, ratio, clip fraction
                with torch.no_grad():
                    kl = (old_log_probs - new_log_probs).mean().abs()
                ratio = (new_log_probs - old_log_probs).exp()
                clipfrac = ((ratio - 1.0).abs() > clip_epsilon).float().mean()

                # Policy loss (clipped)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss: PPO2-style value clipping + Huber
                values_old = old_values.detach()
                values_clipped = values_old + (values - values_old).clamp(-value_clip_range, value_clip_range)
                v_loss1 = F.smooth_l1_loss(values,         returns, reduction='none')
                v_loss2 = F.smooth_l1_loss(values_clipped, returns, reduction='none')
                value_loss = torch.max(v_loss1, v_loss2).mean()

                loss = policy_loss + value_coef * value_loss - ent_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=0.5)
                optimizer.step()

                epoch_loss_sum += loss.item()
                epoch_batches  += 1

                # CSV diagnostics
                log_train_metrics_to_csv(
                    out_dir=log_dir,
                    epoch=epoch,
                    batch_idx=b_idx,
                    old_log_probs=old_log_probs,
                    new_log_probs=new_log_probs,
                    dist_entropy=entropy,
                    advantages=advantages,
                    returns=returns,
                    values=values,
                    clip_eps=clip_epsilon
                )

                if (b_idx % log_every == 0) or (b_idx == 0):
                    print(
                        f"[train-gnn] KL={kl.item():.4f}  clipfrac={clipfrac.item():.3f}  "
                        f"ent={entropy.mean().item():.3f} (coef={ent_coef:.4f})  "
                        f"adv_std={advantages.std().item():.3f}  "
                        f"v_loss={value_loss.item():.4f}  pi_loss={policy_loss.item():.4f}  "
                        f"TOTAL={loss.item():.4f}  v_coef={value_coef:.3f}"
                    )

                # early stop in this chunk if KL too large
                if kl.item() > kl_threshold:
                    print(f"Early stop PPO epochs in this chunk: KL={kl.item():.4f} > {kl_threshold}")
                    break

            # LR scheduler per-epoch
            scheduler.step()
            cur_lr = scheduler.get_last_lr()[0]
            avg_epoch_loss = epoch_loss_sum / max(1, epoch_batches)
            print(f"[chunk {chunk_start//update_batch_size_size + 1} | PPO Epoch {epoch+1}] "
                  f"Avg Loss: {avg_epoch_loss:.4f} | LR: {cur_lr:.6f}")

        # free memory
        chunk_buffer.clear()
        gc.collect()
        torch.cuda.empty_cache()

    # ---- summary ----
    avg_makespan = sum(all_makespans) / max(1, len(all_makespans))
    print("\n=== [Training Summary] ===")
    print(f"Total Episodes: {total_episodes}")
    print(f"Average Makespan: {avg_makespan:.2f}")

    return avg_makespan, avg_epoch_loss if (locals().get('avg_epoch_loss') is not None) else 0.0
