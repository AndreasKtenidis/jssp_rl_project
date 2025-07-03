# jssp_rl/train/train_loop.py

import torch
import time
from train.episode_runner import run_episode
from env.jssp_environment import JSSPEnvironment

def train_loop(dataloader, gnn, actor, critic, edge_index, edge_weights, optimizer, epochs=50):
    gnn.train()
    actor.train()
    critic.train()
    
    episode_makespans = []
    best_makespan = float("inf")
    epoch_losses = []

    for epoch in range(epochs):
        start_time = time.time()
        print(f"* Starting Epoch {epoch+1}/{epochs}")
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            print(f"$  Batch {batch_idx + 1}/{len(dataloader)}")
            optimizer.zero_grad()
            batch_loss = 0

            for i in range(len(batch['times'])):
                instance = {
                    "times": batch['times'][i],
                    "machines": batch['machines'][i]
                }
                env = JSSPEnvironment(instance['times'], instance['machines'])
                log_probs, values, rewards = run_episode(env, gnn, actor, critic, edge_index, edge_weights)

                _, _, _, makespan = env.step(0)
                print(f"**  Instance {i+1}/{len(batch['times'])} | Makespan: {makespan:.2f} | Steps: {len(rewards)}")
                if makespan < best_makespan:
                    best_makespan = makespan

                returns = []
                G = 0
                for r in reversed(rewards):
                    G = r + 0.99 * G
                    returns.insert(0, G)
                returns = torch.stack(returns)

                values = torch.stack(values).squeeze()
                log_probs = torch.stack(log_probs)

                advantages = returns - values.detach()
                policy_loss = -(log_probs * advantages).sum()
                value_loss = advantages.pow(2).mean()

                loss = policy_loss + value_loss
                batch_loss += loss

            batch_loss /= len(batch)
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            print(f"*** Batch Loss: {batch_loss.item():.4f}")

        episode_makespans.append(best_makespan)
        avg_epoch_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        end_time = time.time()

        print(f" # Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f}")
        print(f" @ Epoch {epoch+1} finished in {end_time - start_time:.2f} seconds")

    return episode_makespans, epoch_losses
