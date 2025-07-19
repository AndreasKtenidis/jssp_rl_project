import torch
import time
import os
from train.episode_runner import run_episode
from env.jssp_environment import JSSPEnvironment

def train_loop(
    dataloader,
    gnn,
    actor,
    critic,
    edge_index,
    edge_weights,
    optimizer,
    epochs=1,
    start_epoch=0,
    best_val_makespan=float("inf"),
    save_path="/home/aktenidis/JSSPprojects/jssp_rl_project/checkpointsV2/best_model.pth",
    validate_fn=None,
    val_dataloader=None,
    device=None
):
    device = device # or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gnn.train().to(device)
    actor.train().to(device)
    critic.train().to(device)

    edge_index = edge_index.to(device)
    edge_weights = edge_weights.to(device)

    episode_makespans = []
    epoch_losses = []

    for epoch in range(start_epoch, start_epoch + epochs):
        start_time = time.time()
        print(f"* Starting Epoch {epoch+1}/{start_epoch + epochs}")
        total_loss = 0
        epoch_best_makespan = float("inf")

        for batch_idx, batch in enumerate(dataloader):
            batch_start = time.time()
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}")
            optimizer.zero_grad()
            batch_loss = 0

            for i in range(len(batch['times'])):
                times = batch['times'][i].to(device)
                machines = batch['machines'][i].to(device)
                instance_loss = 0.0

                for episode_idx in range(3):  # Run 3 episodes per instance
                    env = JSSPEnvironment(times, machines, device=device)  # env runs on CPU
                    log_probs, values, rewards = run_episode(env, gnn, actor, critic, edge_index, edge_weights,device=device)

                    makespan = env.get_makespan()

                    print(f"**  Instance {i+1}/{len(batch['times'])} | Episode {episode_idx+1}/3 | Makespan: {makespan:.2f} | Steps: {len(rewards)}")

                    if makespan < epoch_best_makespan:
                        epoch_best_makespan = makespan

                    returns = []
                    G = 0
                    for r in reversed(rewards):
                        G = r + 0.99 * G
                        returns.insert(0, G)
                    returns = torch.stack(returns).to(device)

                    values_tensor = torch.stack(values).squeeze().to(device)
                    log_probs_tensor = torch.stack(log_probs).to(device)

                    advantages = returns - values_tensor.detach()
                    policy_loss = -(log_probs_tensor * advantages).sum()
                    value_loss = advantages.pow(2).mean()

                    instance_loss += (policy_loss + value_loss)

                instance_loss /= 3.0
                batch_loss += instance_loss

            batch_loss /= len(batch)
            batch_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            total_loss += batch_loss.item()
            print(f"*** Batch Loss: {batch_loss.item():.4f}")
            batch_end = time.time()
            print(f"    ⏱️ Batch {batch_idx+1} time: {batch_end - batch_start:.2f}s")

        avg_epoch_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        episode_makespans.append(epoch_best_makespan)

        end_time = time.time()
        print(f" Epoch {epoch+1} - Loss: {avg_epoch_loss:.4f}")
        print(f" Epoch finished in {end_time - start_time:.2f} seconds")

        if validate_fn and val_dataloader:
            val_makespan = validate_fn(val_dataloader, gnn, actor, critic, edge_index, edge_weights, device=device)


            if val_makespan < best_val_makespan:
                best_val_makespan = val_makespan
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'gnn_state_dict': gnn.state_dict(),
                    'actor_state_dict': actor.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_makespan': best_val_makespan
                }, save_path)
                print(f"  ✅ Saved new best model at epoch {epoch+1} with val makespan {val_makespan:.2f}")

    return episode_makespans, epoch_losses
