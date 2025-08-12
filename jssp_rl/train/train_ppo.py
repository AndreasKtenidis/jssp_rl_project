import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Data
import gc

from utils.buffer import RolloutBuffer
from utils.features import prepare_features
from models.gnn import GNNWithAttention
from env.jssp_environment import JSSPEnvironment


from config import batch_size, epochs, gamma, gae_lambda, clip_epsilon, value_coef, entropy_coef

def train(dataloader, actor_critic, optimizer, device):
    actor_critic.train()
    all_makespans = []
    global_buffer = RolloutBuffer()

    print("\n=== [1] Collecting all trajectories with θ_old ===")
    total_episodes = 0

    for batch_idx, batch in enumerate(dataloader):
        times_batch = batch['times']
        machines_batch = batch['machines']
        current_batch_size = times_batch.shape[0]

        for i in range(current_batch_size):
            times = times_batch[i]
            machines = machines_batch[i]

            env = JSSPEnvironment(times, machines)
            env.reset()
            done = False
            episode_reward = 0
            buffer = RolloutBuffer()

            while not done:
                edge_index = GNNWithAttention.build_edge_index_with_machine_links(machines)
                x = prepare_features(env, edge_index, device)
                data = Data(x=x, edge_index=edge_index.to(device))

                available = env.get_available_actions()
                mask = torch.zeros(x.size(0), dtype=torch.bool, device=device)
                mask[available] = True

                with torch.no_grad():
                    action, log_prob, value = actor_critic.act(data, mask=mask)

                _, reward, done, _ = env.step(action)
                episode_reward += reward
                buffer.add(data, action, reward, log_prob, value, done)

            # Final state value (bootstrap)
            with torch.no_grad():
                final_edge_index = GNNWithAttention.build_edge_index_with_machine_links(machines)
                final_x = prepare_features(env, final_edge_index, device)
                final_data = Data(x=final_x, edge_index=final_edge_index.to(device))
                _, final_value = actor_critic(final_data)

            buffer.compute_returns_and_advantages(final_value.squeeze(), gamma, gae_lambda)
            global_buffer.merge(buffer)

            makespan = env.get_makespan()
            all_makespans.append(makespan)
            total_episodes += 1

            print(f"[Batch {batch_idx + 1} | Ep {total_episodes}] Makespan: {makespan:.2f} | Total Reward: {episode_reward:.2f}")

    print(f"\n=== [2] PPO Update over {total_episodes} collected episodes ===")

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        print(f"\n--- PPO Epoch {epoch + 1}/{epochs} ---")

        for batch_idx, (states, actions, old_log_probs, returns, advantages) in enumerate(global_buffer.get_batches(batch_size)):
            # Move tensors to correct device
            states = states.to(device)
            actions = actions.to(device)
            old_log_probs = old_log_probs.to(device)
            returns = returns.to(device)
            advantages = advantages.to(device)

            # Forward pass
            action_logits, values = actor_critic(states)
            dist = Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # PPO loss components
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values.squeeze(-1), returns.squeeze(-1))
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 50 == 0 or batch_idx == num_batches - 1:
                print(f"  [Batch {batch_idx + 1}] "
                    f"Loss: {loss.item():.4f} | "
                    f"Policy: {policy_loss.item():.4f} | "
                    f"Value: {value_loss.item():.4f} | "
                    f"Entropy: {entropy.item():.4f}")

        avg_epoch_loss = epoch_loss / num_batches
        print(f"[Epoch {epoch + 1}] ✅ Avg PPO Loss: {avg_epoch_loss:.4f} over {num_batches} batches")


    avg_makespan = sum(all_makespans) / len(all_makespans)
    print("\n=== [Training Summary] ===")
    print(f"Total Episodes: {total_episodes}")
    print(f"Average Makespan: {avg_makespan:.2f}")

    global_buffer.clear()
    del global_buffer
    gc.collect()
    torch.cuda.empty_cache()
    return avg_makespan, avg_epoch_loss

