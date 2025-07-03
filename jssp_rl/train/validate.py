# jssp_rl/train/validate.py

import torch
from train.episode_runner import run_episode
from env.jssp_environment import JSSPEnvironment

def validate(dataloader, gnn, actor, critic, edge_index, edge_weights):
    gnn.eval()
    actor.eval()
    critic.eval()

    total_makespan = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            for i in range(len(batch['times'])):
                instance = {
                    "times": batch['times'][i],
                    "machines": batch['machines'][i]
                }
                env = JSSPEnvironment(instance['times'], instance['machines'])
                log_probs, values, rewards = run_episode(env, gnn, actor, critic, edge_index, edge_weights, epsilon=0.0)
                _, _, _, makespan = env.step(0)  # or use env.job_completion_times.max()
                total_makespan += makespan
                count += 1

    avg_makespan = total_makespan / count
    print(f"🧪 Validation Avg Makespan: {avg_makespan:.2f}")
    return avg_makespan
