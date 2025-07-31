import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Data

from utils.buffer import RolloutBuffer
from utils.features import prepare_features
from models.gnn import GNNWithAttention
from config import clip_epsilon, gamma, gae_lambda, num_epochs, value_coef, entropy_coef
from env.jssp_environment import JSSPEnvironment
import time, psutil

def train(dataloader, actor_critic, optimizer, device):
    actor_critic.train()
    all_makespans = []
    total_loss = 0.0
    episode_count = 0

    for batch_idx, batch in enumerate(dataloader):
        print(f"      ↳ inner step start  {time.strftime('%H:%M:%S')}  "
             f"cpu {psutil.cpu_percent():.0f}%", flush=True)
        print(f"\n--- Training Batch {batch_idx + 1} ---")
        batch_loss = 0.0
        buffer = RolloutBuffer()

        times_batch = batch['times']
        machines_batch = batch['machines']
        batch_size = times_batch.shape[0]

        for i in range(batch_size):
            times = times_batch[i]
            machines = machines_batch[i]

            env = JSSPEnvironment(times.to(device), machines.to(device), device=device)
            env.reset()

            done = False
            episode_reward = 0
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

            with torch.no_grad():
                final_edge_index = GNNWithAttention.build_edge_index_with_machine_links(machines)
                final_x = prepare_features(env, final_edge_index, device)
                final_data = Data(x=final_x, edge_index=final_edge_index.to(device))
                _, final_value = actor_critic(final_data)
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
                
                final_value = final_value.to(device)


            buffer.compute_returns_and_advantages(final_value.squeeze(), gamma, gae_lambda)

            
            # Optimize policy
            for epoch in range(num_epochs):
                for states, actions, old_log_probs, returns, advantages in buffer.get_batches(batch_size):
                    action_logits, values = actor_critic(states)
                    # actions      = actions.to(device)
                    # old_log_probs = old_log_probs.to(device)
                    # returns       = returns.to(device)
                    # advantages    = advantages.to(device)
                
                    if returns.numel() > 1:
                        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-6)
                    else:
                        returns = torch.zeros_like(returns)

                    returns = returns * 0.1

                    if advantages.numel() > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
                    else:
                        advantages = torch.zeros_like(advantages)
                        
                    if torch.isnan(action_logits).any():
                        print("❌ NaNs in action_logits!", action_logits)
                        raise ValueError("NaNs in logits – check input tensors or network weights.")

                    dist = Categorical(logits=action_logits)

                    new_log_probs = dist.log_prob(actions)
                    entropy = dist.entropy().mean()

                    ratio = (new_log_probs - old_log_probs).exp()
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.MSELoss()(values.squeeze(-1), returns.squeeze(-1))
                    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_loss += loss.item()

            makespan = env.get_makespan()
            all_makespans.append(makespan)
            buffer.clear()
            episode_count += 1
            print(f"[Episode {episode_count}] Makespan: {makespan:.2f}, Total Reward: {episode_reward:.2f}")

        total_loss += batch_loss
        avg_batch_loss = batch_loss / batch_size
        print(f"Finished Batch {batch_idx + 1} | Avg Batch Loss: {avg_batch_loss:.4f}")

    avg_makespan = sum(all_makespans) / len(all_makespans)
    avg_loss = total_loss / len(all_makespans)

    print(f"\n--- End of Epoch ---")
    print(f"Average Makespan: {avg_makespan:.2f}")
    print(f"Average Loss: {avg_loss:.4f}")

    return avg_makespan, avg_loss
