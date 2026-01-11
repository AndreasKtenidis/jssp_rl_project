import gc 
import os 
import torch 
import torch.nn.functional as F 

from torch.optim.lr_scheduler import CosineAnnealingLR 
from torch.distributions import Categorical 
from utils.HeteroBuffer import RolloutBuffer 
from utils.Heterofeatures import prepare_features 
from utils.action_masking import select_top_k_actions
from env.jssp_environment import JSSPEnvironment 
from utils.train_diagnostics import log_train_metrics_to_csv 
from utils.policy_utils import per_graph_logprob_and_entropy
from utils.running_norm import RunningReturnNormalizer

from config import gae_lambda,gamma,lr,epochs, entropy_coef,batch_size,clip_epsilon,value_coef

def train(
    dataloader,
    actor_critic,
    optimizer,
    device,
    update_batch_size_size: int = 16,
    value_clip_range: float = 0.2,   # PPO2-style value clipping (RAW space)
    kl_threshold: float = 0.25,      # early stop per chunk
    log_every: int = 50,
    log_dir=None,
):
    """
    Clean PPO baseline:
      • Rollout: actor_critic.eval(), Categorical over masked logits, NO heuristics.
      • Update: actor_critic.train(), SAME masking as rollout.
      • Values/returns: no normalization; RAW value clipping only.
      • Normalize ONLY advantages (per kept graphs).
      • Ignore top-k artifacts; keep forced steps out of policy loss via 'keep'.
    """
    if log_dir is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))  # .../jssp_rl
        log_dir = os.path.join(base_dir, "outputs", "logs")

    all_makespans = []
    total_episodes = 0

    print("\n=== PPO (HeteroGIN) Training in update-batches ===")

    dataset = dataloader.dataset if hasattr(dataloader, "dataset") else dataloader
    # running normalizer for returns (used only in value loss)
    return_norm = RunningReturnNormalizer()

    for chunk_start in range(0, len(dataset), update_batch_size_size):
        chunk = dataset[chunk_start:chunk_start + update_batch_size_size]
        chunk_buffer = RolloutBuffer()

        # Cosine LR per-chunk
        base_lr = optimizer.param_groups[0]['lr']
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * 0.01)
        

        # =========================
        # 1) Rollout collection (EVAL mode → stable logits/Dropout off)
        # =========================
        actor_critic.eval()

        for instance in chunk:
            times    = instance["times"] #why? in validate we dont
            machines = instance["machines"]

            env = JSSPEnvironment(times, machines)
            env.reset()

            step_idx = 0
            done = False
            ep_reward = 0.0
            ep_buf = RolloutBuffer()
            one_legal_counter = 0  # diagnostics

            while not done:
                # Build graph features
                data = prepare_features(env, device)  # HeteroData (on device)
                with torch.no_grad():
                    logits, value = actor_critic(data)  # action logits per op, scalar value per graph

                if torch.isnan(logits).any():
                    raise ValueError("NaNs in logits – check inputs/weights.")

                num_ops = int(data["op"].x.size(0))

                # Get legal mask ONLY (ignore any "top-k" info)
                valid_idx, valid_mask, _topk_idx, _topk_mask = select_top_k_actions(
                    env, top_k=1, device=device
                )

                # First-step consistency check (env.get_available_actions() ≈ one per job)
                if step_idx == 0:
                    legal_mask = int(valid_mask.sum().item())
                    print(
                        f"[episode {total_episodes + 1} | first step] "
                        f"mask_legal={legal_mask}  num_jobs={env.num_jobs}  num_machines={env.num_machines}"
                    )
                    assert legal_mask == env.num_jobs, (
                        f"At first step, expected {env.num_jobs} legal actions (one per job), "
                        f"but got {legal_mask}. Check index alignment between env and features."
                    )
                    assert valid_mask.numel() == data['op'].x.size(0), ( #otherwise index misalignment
                        f"Mask length ({valid_mask.numel()}) != #op nodes ({data['op'].x.size(0)})."
                    )

                legal_count = int(valid_mask.sum().item())
                if legal_count == 0:
                    # No legal action → ill-posed state; stop this episode safely
                    print(f"[ROLL] No-legal-actions at t={step_idx} → abort episode")
                    break
                if legal_count == 1:
                    one_legal_counter += 1
                    print(f"[ROLL] One-legal-action at t={step_idx} | mask_legal=1")

                # Hard-mask the logits with valid_mask
                masked_logits = logits.clone().masked_fill(~valid_mask, -1e10)

                # === Action selection (SAME distribution at update) ===
                dist = Categorical(logits=masked_logits)
                action_t = dist.sample()
                action = int(action_t.item())
                log_prob = dist.log_prob(action_t)  # store old log-prob from EXACT rollout policy

                # Step the env
                _, reward, done, _ = env.step(action)
                ep_reward += reward
                step_idx += 1

                # Store transition (mask only; keep topk_mask as all False for compatibility)
                topk_mask_false = torch.zeros_like(valid_mask, dtype=torch.bool, device=valid_mask.device)
                ep_buf.add(
                    state=data,
                    action=action,
                    reward=reward,
                    log_prob=log_prob,
                    value=value.view(-1),  # keep tensor shape consistent
                    done=done,
                    mask=valid_mask,
                    topk_mask=topk_mask_false
                )

            # bootstrap value
            with torch.no_grad():
                final_data = prepare_features(env, device)
                _, final_value = actor_critic(final_data)

            ep_buf.compute_returns_and_advantages(final_value.squeeze(), gamma, gae_lambda)
            chunk_buffer.merge(ep_buf)

            mk = env.get_makespan()
            all_makespans.append(mk)
            total_episodes += 1
            print(
                f"[chunk {chunk_start//update_batch_size_size + 1} | Ep {total_episodes}] "
                f"Makespan: {mk:.2f} | Total Reward: {ep_reward:.2f} | "
                f"one-legal-occurrences={one_legal_counter}"
            )

        print(f"\n=== PPO Update for chunk starting at {chunk_start} ===")

        # =========================
        # 2) PPO update (clean baseline)
        # =========================
        last_epoch_avg_loss = 0.0
        actor_critic.train()
        ent_coef = entropy_coef  # constant (no decay)

        for epoch in range(epochs):
            epoch_loss_sum = 0.0
            epoch_batches  = 0
            epoch_weight_sum = 0

            for b_idx, batch in enumerate(chunk_buffer.get_batches(batch_size)):
                # Buffer may return with/without topk_masks (we ignore them anyway)
                if len(batch) == 8:
                    states, actions, old_log_probs, returns, advantages, old_values, masks, _topk_masks = batch
                else:
                    states, actions, old_log_probs, returns, advantages, old_values, masks = batch

                # — move to device —
                actions       = actions.to(device)
                old_log_probs = old_log_probs.to(device)
                returns       = returns.to(device)
                advantages    = advantages.to(device)
                old_values    = old_values.to(device)
                masks         = masks.to(device)
                states        = states.to(device) if hasattr(states, "to") else states

                # Forward
                logits, values = actor_critic(states)

                # Apply EXACT SAME legal mask as rollout
                logits = logits.masked_fill(~masks, -1e10)

                # ==== per-graph bookkeeping ====
                batch_vec = states['op'].batch.view(-1)
                B = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 0
                sizes = torch.bincount(batch_vec, minlength=B).tolist()
                legal_counts = [int((masks[batch_vec == g]).sum().item()) for g in range(B)]
                try:
                    print(f"[upd] graphs={B} | nodes/graph={sizes} | legal/graph={legal_counts} | actions={actions.tolist()}")
                except Exception:
                    print(f"[upd] graphs={B} | nodes/graph={sizes} | legal/graph={legal_counts}")

                # Keep graphs with ≥2 legal actions (avoid forced-steps in policy loss)
                keep = torch.tensor([c >= 2 for c in legal_counts], device=device, dtype=torch.bool)
                num_kept = int(keep.sum().item())
                if num_kept == 0:
                    print("[upd] Skipping batch: no graphs with legal>=2")
                    continue
                denom = keep.sum().float()  # normalize by #kept graphs

                # New log-probs & entropy per-graph (vectorized via helper)
                new_log_probs, ent_vec = per_graph_logprob_and_entropy(
                    logits.view(-1), states['op'].batch, actions
                )

                # Filtered views for logs/stats
                k_idx   = keep.nonzero(as_tuple=False).view(-1)
                old_lp_k = old_log_probs[k_idx]
                new_lp_k = new_log_probs[k_idx]

                # === Normalize ONLY advantages (per kept subset) ===
                if k_idx.numel() > 1:
                    adv_k = advantages[k_idx]
                    adv_k = (adv_k - adv_k.mean()) / (adv_k.std(unbiased=False) + 1e-8)
                    advantages = advantages.clone()
                    advantages[k_idx] = adv_k

                # Ratios & stats (kept only)
                ratio   = (new_log_probs - old_log_probs).exp()
                ratio_k = ratio[k_idx]
                r_mean  = ratio_k.mean().item()
                r_std   = ratio_k.std(unbiased=False).item() if ratio_k.numel() > 1 else 0.0
                r_min   = ratio_k.min().item()
                r_max   = ratio_k.max().item()

                # KL & clipfrac (kept)
                kl = ((old_log_probs - new_log_probs).abs()[k_idx]).mean()
                clipfrac = (((ratio_k - 1.0).abs() > clip_epsilon).float().mean())

                # === Value loss in RAW space with PPO2-style clipping ===
                values_raw     = values.view(-1)
                old_values_raw = old_values.view(-1).detach()
                returns_raw    = returns.view(-1).detach()

                values_clipped_raw = old_values_raw + (values_raw - old_values_raw).clamp(
                    -value_clip_range, value_clip_range
                )

                # Update running stats & build normalized versions ONLY for the loss
                return_norm.update(returns_raw)                        # update stats
                ret_z   = return_norm.normalize(returns_raw)           # target (z-score)
                val_z   = return_norm.normalize(values_raw)            # prediction in same z-space
                vclip_z = return_norm.normalize(values_clipped_raw)    # clipped prediction in z-space

                # Huber loss σε normalized space 
                v_loss1 = F.smooth_l1_loss(val_z,   ret_z, reduction='none')
                v_loss2 = F.smooth_l1_loss(vclip_z, ret_z, reduction='none')
                value_loss = (torch.max(v_loss1, v_loss2)[k_idx].sum() / denom)

                # Raw-space RMSE for monitoring 
                with torch.no_grad():
                    rmse_value_raw = torch.sqrt(F.mse_loss(values_raw[k_idx], returns_raw[k_idx]))

                # === Policy loss (clipped surrogate) ===
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
                pi_vec = -torch.min(surr1, surr2)
                policy_loss = (pi_vec[k_idx].sum() / denom)

                # === Entropy (weighted mean) ===
                entropy = (ent_vec[k_idx].sum() / denom)

                loss = policy_loss + value_coef * value_loss - ent_coef * entropy

                # Safety checks
                for name, t in [("policy_loss", policy_loss), ("value_loss", value_loss),
                                ("entropy", entropy), ("loss", loss)]:
                    if not torch.isfinite(t):
                        print(f"[WARN] Non-finite {name}: {t}")
                        print(f"  returns.mean={returns_raw.mean().item():.4f} std={returns_raw.std(unbiased=False).item():.4f}")
                        print(f"  values.mean={values_raw.mean().item():.4f}  std={values_raw.std(unbiased=False).item():.4f}")
                        print(f"  adv.mean={advantages.mean().item():.4f} std={advantages.std(unbiased=False).item():.4f}")
                        break

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=0.5)
                optimizer.step()

                epoch_loss_sum   += float(loss.detach().item()) * int(denom.item())
                epoch_weight_sum += int(denom.item())
                epoch_batches    += 1

                # Logging (log only kept samples to be consistent)
                log_train_metrics_to_csv(
                    out_dir=log_dir,
                    epoch=epoch,
                    batch_idx=b_idx,
                    old_log_probs=old_lp_k,
                    new_log_probs=new_lp_k,
                    dist_entropy=entropy,
                    advantages=advantages[k_idx],
                    returns=returns_raw[k_idx],   # raw for interpretability
                    values=values_raw[k_idx],     # raw for interpretability
                    clip_eps=clip_epsilon,
                )

                valid_min = logits[masks].min().item()   # only valid positions
                valid_max = logits[masks].max().item()
                
                if (b_idx % log_every == 0) or (b_idx == 0):
                    print(
                        f"logits(valid) min/max: {valid_min:.2f}/{valid_max:.2f} | "
                        f"[train-Hetero] KL={kl.item():.4f}  clipfrac={clipfrac.item():.3f}  "
                        f"ratio mean/std/min/max={r_mean:.3f}/{r_std:.3f}/{r_min:.3f}/{r_max:.3f}  "
                        f"ent={entropy.item():.3f} (coef={ent_coef:.4f})  "
                        f"adv_std={advantages[k_idx].std(unbiased=False).item() if num_kept>1 else 0.0:.3f}  "
                        f"v_loss={value_loss.item():.4f}  pi_loss={policy_loss.item():.4f}  "
                        f"TOTAL={loss.item():.4f}  v_coef={value_coef:.3f}  "
                        f"policy_samples={num_kept}/{B}  "
                        f"raw_RMSE={rmse_value_raw.item():.3f}"
                    )

                # Early stop per chunk
                if kl.item() > kl_threshold:
                    print(f"⛳ Early stop PPO epochs in this chunk: KL={kl.item():.4f} > {kl_threshold}")
                    break

            # Cosine LR step per epoch
            scheduler.step()
            cur_lr = scheduler.get_last_lr()[0]

            if epoch_weight_sum > 0:
                avg_epoch_loss = epoch_loss_sum / epoch_weight_sum
            else:
                avg_epoch_loss = float('nan')

            last_epoch_avg_loss = avg_epoch_loss
            print(
                f"[chunk {chunk_start//update_batch_size_size + 1} | PPO Epoch {epoch+1}] "
                f"Avg Loss: {avg_epoch_loss:.4f} | LR: {cur_lr:.6f}"
            )

        # cleanup
        chunk_buffer.clear()
        gc.collect()
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()

    avg_mk = sum(all_makespans) / max(1, len(all_makespans))
    print("\n=== [Training Summary] ===")
    print(f"Total Episodes: {total_episodes}")
    print(f"Average Makespan: {avg_mk:.2f}")

    return avg_mk, last_epoch_avg_loss
