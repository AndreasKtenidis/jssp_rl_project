import torch
from train.episode_runner import run_episode
from env.jssp_environment import JSSPEnvironment

def validate(dataloader, gnn, actor, critic, edge_index, edge_weights, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gnn = gnn.to(device).eval()
    actor = actor.to(device).eval()
    critic = critic.to(device).eval()

    edge_index = edge_index.to(device)
    edge_weights = edge_weights.to(device)

    total_makespan = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            for i in range(len(batch['times'])):
                times = batch['times'][i]  
                machines = batch['machines'][i]  
                env = JSSPEnvironment(times, machines, device=device)

                run_episode(
                    env, gnn, actor, critic,
                    edge_index, edge_weights,
                    epsilon=0.0,
                    device=device  
                )

                makespan = env.get_makespan()
  
                total_makespan += makespan
                count += 1

    avg_makespan = total_makespan / count
    print(f"**** Validation Avg Makespan: {avg_makespan:.2f}")
    return avg_makespan
