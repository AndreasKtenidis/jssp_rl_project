import gc 
import os 
import torch 
import torch.nn.functional as F 
from torch.optim.lr_scheduler import CosineAnnealingLR 

from utils.HeteroBuffer import RolloutBuffer 
from env.jssp_environment import JSSPEnvironment 
from utils.train_diagnostics import log_train_metrics_to_csv 
from utils.policy_utils import per_graph_logprob_and_entropy
from utils.running_norm import RunningReturnNormalizer
from utils.rollout_once import rollout_once   


from config import gae_lambda, gamma, lr, epochs, entropy_coef, batch_size, clip_epsilon, value_coef


def train(
    dataloader,
    actor_critic,
    optimizer,
    device,
    use_est_boost, 
    est_beta, 
    update_batch_size: int = 16,
    value_clip_range: float = 0.2,   # PPO2-style value clipping
    kl_threshold: float = 0.65,      # early stop per chunk
    log_every: int = 50,
    log_dir=None,
       
):
    if log_dir is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))  # .../jssp_rl
        log_dir = os.path.join(base_dir, "outputs", "logs")

    all_makespans = []
    total_episodes = 0

    print("\n=== PPO (HeteroGIN) Training in update-batches ===")

    dataset = dataloader.dataset if hasattr(dataloader, "dataset") else dataloader
    return_norm = RunningReturnNormalizer()

    for chunk_start in range(0, len(dataset), update_batch_size):
        chunk = dataset[chunk_start:chunk_start + update_batch_size]
        chunk_buffer = RolloutBuffer()

        # Cosine LR per-chunk
        base_lr = optimizer.param_groups[0]['lr']
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * 0.01)

        # =========================
        # 1) Rollout collection
        # =========================
        for instance in chunk:
            env = JSSPEnvironment(instance["times"], instance["machines"])

            # unified rollout (collects buffer + stats)
            ep_buf, stats = rollout_once(
                env,
                actor_critic,
                device,
                use_est_boost=use_est_boost,
                est_beta=est_beta,
                mode="stochastic"
                
            )

            chunk_buffer.merge(ep_buf)

            mk = stats["makespan"]
            all_makespans.append(mk)
            total_episodes += 1
            print(
                f"[chunk {chunk_start//update_batch_size + 1} | Ep {total_episodes}] "
                f"Makespan: {mk:.2f} | Total Reward: {stats['total_reward']:.2f} | "
                f"one-legal-occurrences={stats['one_legal_counter']}"
            )

        print(f"\n=== PPO Update for chunk starting at {chunk_start} ===")

        # =========================
        # 2) PPO update
        # =========================
        last_epoch_avg_loss = 0.0
        actor_critic.train()
        ent_coef = entropy_coef

        for epoch in range(epochs):
            epoch_loss_sum = 0.0
            epoch_batches = 0
            epoch_weight_sum = 0

            for b_idx, batch in enumerate(chunk_buffer.get_batches(batch_size)):
                if len(batch) == 8:
                    states, actions, old_log_probs, returns, advantages, old_values, masks, _ = batch
                else:
                    states, actions, old_log_probs, returns, advantages, old_values, masks = batch

                # to device
                actions       = actions.to(device)
                old_log_probs = old_log_probs.to(device)
                returns       = returns.to(device)
                advantages    = advantages.to(device)
                old_values    = old_values.to(device)
                masks         = masks.to(device)
                states        = states.to(device) if hasattr(states, "to") else states

                # Forward pass
                logits, values = actor_critic(states)
                logits = logits.masked_fill(~masks, -1e10)

                # bookkeeping
                batch_vec = states['op'].batch.view(-1)
                B = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 0
                sizes = torch.bincount(batch_vec, minlength=B).tolist()
                legal_counts = [int((masks[batch_vec == g]).sum().item()) for g in range(B)]
                print(f"[upd] graphs={B} | nodes/graph={sizes} | legal/graph={legal_counts}")

                # filter graphs
                keep = torch.tensor([c >= 2 for c in legal_counts], device=device, dtype=torch.bool)
                if keep.sum() == 0:
                    print("[upd] Skipping batch: no graphs with legal>=2")
                    continue
                denom = keep.sum().float()

                # log_probs + entropy
                new_log_probs, ent_vec = per_graph_logprob_and_entropy(
                    logits.view(-1), states['op'].batch, actions
                )

                k_idx = keep.nonzero(as_tuple=False).view(-1)
                old_lp_k = old_log_probs[k_idx]
                new_lp_k = new_log_probs[k_idx]

                # normalize advantages only on kept
                if k_idx.numel() > 1:
                    adv_k = advantages[k_idx]
                    adv_k = (adv_k - adv_k.mean()) / (adv_k.std(unbiased=False) + 1e-8)
                    advantages = advantages.clone()
                    advantages[k_idx] = adv_k

                # Ratios + stats
                ratio = (new_log_probs - old_log_probs).exp()
                ratio_k = ratio[k_idx]
                kl = ((old_log_probs - new_log_probs).abs()[k_idx]).mean()
                clipfrac = (((ratio_k - 1.0).abs() > clip_epsilon).float().mean())

                # Value loss with clipping
                values_raw = values.view(-1)
                old_values_raw = old_values.view(-1).detach()
                returns_raw = returns.view(-1).detach()

                values_clipped_raw = old_values_raw + (values_raw - old_values_raw).clamp(
                    -value_clip_range, value_clip_range
                )

                return_norm.update(returns_raw)
                ret_z = return_norm.normalize(returns_raw)
                val_z = return_norm.normalize(values_raw)
                vclip_z = return_norm.normalize(values_clipped_raw)

                v_loss1 = F.smooth_l1_loss(val_z, ret_z, reduction='none')
                v_loss2 = F.smooth_l1_loss(vclip_z, ret_z, reduction='none')
                value_loss = (torch.max(v_loss1, v_loss2)[k_idx].sum() / denom)

                with torch.no_grad():
                    rmse_value_raw = torch.sqrt(F.mse_loss(values_raw[k_idx], returns_raw[k_idx]))

                # Policy loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
                pi_vec = -torch.min(surr1, surr2)
                policy_loss = (pi_vec[k_idx].sum() / denom)

                # Entropy
                entropy = (ent_vec[k_idx].sum() / denom)

                loss = policy_loss + value_coef * value_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=0.5)
                optimizer.step()

                epoch_loss_sum += float(loss.detach().item()) * int(denom.item())
                epoch_weight_sum += int(denom.item())
                epoch_batches += 1

                log_train_metrics_to_csv(
                    out_dir=log_dir,
                    epoch=epoch,
                    batch_idx=b_idx,
                    old_log_probs=old_lp_k,
                    new_log_probs=new_lp_k,
                    dist_entropy=entropy,
                    advantages=advantages[k_idx],
                    returns=returns_raw[k_idx],
                    values=values_raw[k_idx],
                    clip_eps=clip_epsilon,
                )

                if (b_idx % log_every == 0) or (b_idx == 0):
                    print(
                        f"[train-Hetero] KL={kl.item():.4f}  clipfrac={clipfrac.item():.3f}  "
                        f"v_loss={value_loss.item():.4f}  pi_loss={policy_loss.item():.4f}  "
                        f"TOTAL={loss.item():.4f}  raw_RMSE={rmse_value_raw.item():.3f}"
                    )

                if kl.item() > kl_threshold:
                    print(f"⛳ Early stop PPO epochs in this chunk: KL={kl.item():.4f} > {kl_threshold}")
                    break

            scheduler.step()
            cur_lr = scheduler.get_last_lr()[0]

            if epoch_weight_sum > 0:
                avg_epoch_loss = epoch_loss_sum / epoch_weight_sum
            else:
                avg_epoch_loss = float('nan')

            last_epoch_avg_loss = avg_epoch_loss
            print(
                f"[chunk {chunk_start//update_batch_size + 1} | PPO Epoch {epoch+1}] "
                f"Avg Loss: {avg_epoch_loss:.4f} | LR: {cur_lr:.6f}"
            )

        chunk_buffer.clear()
        gc.collect()
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()

    avg_mk = sum(all_makespans) / max(1, len(all_makespans))
    print("\n=== [Training Summary] ===")
    print(f"Total Episodes: {total_episodes}")
    print(f"Average Makespan: {avg_mk:.2f}")

    return avg_mk, last_epoch_avg_loss