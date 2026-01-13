import torch
from torch.distributions import Categorical
from torch_geometric.data import Data
import gc
import torch.nn.functional as F

from utils.buffer import RolloutBuffer
from utils.features import prepare_features
from models.gnn import GNNWithAttention
from env.jssp_environment import JSSPEnvironment
from utils.action_masking import select_top_k_actions

from torch.optim.lr_scheduler import CosineAnnealingLR  
from config import batch_size, epochs, gamma, gae_lambda, clip_epsilon, value_coef, entropy_coef, lr  # <- ensure lr is imported

def train(dataloader, actor_critic, optimizer, device, switch_epoch=7, update_batch_size_size=32):
    actor_critic.train()
    all_makespans = []
    total_episodes = 0

    print("\n=== PPO Training in update_batch_sizes ===")

    dataset = dataloader.dataset if hasattr(dataloader, "dataset") else dataloader

    for update_batch_size_start in range(0, len(dataset), update_batch_size_size):
        update_batch_size = dataset[update_batch_size_start:update_batch_size_start + update_batch_size_size]
        update_batch_size_buffer = RolloutBuffer()

        # Cosine LR per chunk  
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

        #  Collect rollouts for this chunk 
        for instance in update_batch_size:
            times = instance['times']
            machines = instance['machines']

            env = JSSPEnvironment(times, machines)
            env.reset()
            done = False
            episode_reward = 0
            buffer = RolloutBuffer()

            while not done:
                edge_index = GNNWithAttention.build_edge_index_with_machine_links(machines)
                x = prepare_features(env, edge_index, device)
                data = Data(x=x, edge_index=edge_index.to(device))

                with torch.no_grad():
                    action_logits, value = actor_critic(data)

                # Guard for NaNs
                if torch.isnan(action_logits).any():
                    print("❌ NaNs detected in action_logits!")
                    print("Logits:", action_logits)
                    raise ValueError("NaNs in logits – check GNN output or invalid input")

                if total_episodes < switch_epoch:
                    # heuristic warmup
                    available_actions, _ = select_top_k_actions(env, top_k=5, device=device)
                    if torch.rand(1).item() < 0.1:
                        action = available_actions[torch.randint(len(available_actions), (1,))].item()
                    else:
                        action = available_actions[0].item()
                    log_prob = torch.log_softmax(action_logits, dim=0)[action]
                else:
                    # learned policy
                    available = env.get_available_actions()
                    mask = torch.zeros(x.size(0), dtype=torch.bool, device=device)
                    mask[available] = True
                    action_logits[~mask] = -1e10
                    dist = Categorical(logits=action_logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                _, reward, done, _ = env.step(action)
                episode_reward += reward
                buffer.add(data, action, reward, log_prob, value, done)

            with torch.no_grad():
                final_edge_index = GNNWithAttention.build_edge_index_with_machine_links(machines)
                final_x = prepare_features(env, final_edge_index, device)
                final_data = Data(x=final_x, edge_index=final_edge_index.to(device))
                _, final_value = actor_critic(final_data)

            buffer.compute_returns_and_advantages(final_value.squeeze(), gamma, gae_lambda)
            update_batch_size_buffer.merge(buffer)

            makespan = env.get_makespan() if hasattr(env, "get_makespann") else env.get_makespan()
            all_makespans.append(makespan)
            total_episodes += 1
            print(f"[update_batch_size {update_batch_size_start//update_batch_size_size + 1} | Ep {total_episodes}] Makespan: {makespan:.2f} | Total Reward: {episode_reward:.2f}")

        print(f"\n=== PPO Update for update_batch_size starting at {update_batch_size_start} ===")

        # ── PPO updates for this chunk ──────────────────────────────────────────
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (states, actions, old_log_probs, returns, advantages) in enumerate(update_batch_size_buffer.get_batches(batch_size)):
                states = states.to(device)
                actions = actions.to(device)
                old_log_probs = old_log_probs.to(device)
                returns = returns.to(device)
                advantages = advantages.to(device)

                # Normalize returns 
                if returns.numel() > 1:
                    returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
                    returns = returns 

                # Normalize advantages 
                if advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

                action_logits, values = actor_critic(states)
                if torch.isnan(action_logits).any():
                    print("❌ NaNs in action_logits!", action_logits)
                    raise ValueError("NaNs in logits  check input tensors or network weights.")

                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.smooth_l1_loss(values.view(-1), returns.view(-1))
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            # ── Cosine LR step  ───────────────────────
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"[update_batch_size {update_batch_size_start//update_batch_size_size + 1} | PPO Epoch {epoch+1}] "
                  f"Avg Loss: {epoch_loss / max(1, num_batches):.4f} | LR: {current_lr:.6f}")

        update_batch_size_buffer.clear()
        gc.collect()
        torch.cuda.empty_cache()

    avg_makespan = sum(all_makespans) / max(1, len(all_makespans))
    print("\n=== [Training Summary] ===")
    print(f"Total Episodes: {total_episodes}")
    print(f"Average Makespan: {avg_makespan:.2f}")

    return avg_makespan, epoch_loss
