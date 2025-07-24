import torch
from torch_geometric.data import Data
from env.jssp_environment import JSSPEnvironment
from models.gnn import GNNWithAttention
from utils.features import prepare_features

@torch.no_grad()
def validate_ppo(dataloader, actor_critic, device=None):
    """
    Greedy argmax rollout with saved policy PPO.
    Used 10 % of synthetic data to validate makespan
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_critic = actor_critic.to(device).eval()

    total_makespan, count = 0.0, 0

    for batch in dataloader:
        for i in range(len(batch['times'])):
            times     = batch['times'][i]
            machines  = batch['machines'][i]

            env = JSSPEnvironment(times, machines, device=device)
            env.reset()
            done  = False

            while not done:
                edge_index = GNNWithAttention.build_edge_index_with_machine_links(machines)
                x = prepare_features(env, edge_index, device)
                data = Data(x=x, edge_index=edge_index.to(device))

                avail = env.get_available_actions()
                mask = torch.zeros(x.size(0), dtype=torch.bool, device=device)
                mask[avail] = True

                action, _, _ = actor_critic.act(data, mask=mask)
                _, _, done, _ = env.step(action)

            total_makespan += env.get_makespan()
            count += 1

    avg_makespan = total_makespan / count
    print(f"**** PPO Validation Avg Makespan: {avg_makespan:.2f}")
    return avg_makespan
