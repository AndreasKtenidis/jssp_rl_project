import torch
from torch.distributions import Categorical
from torch_geometric.data import Data
from env.jssp_environment import JSSPEnvironment
from utils.features import make_hetero_data
# from models.gnn import GNNWithAttention

@torch.no_grad()
def validate_ppo(
    dataloader,
    actor_critic,
    device,
    *,
    limit_instances: int = 50,   # cap validation size
    best_of_k: int = 1,          # >1 -> K stochastic rollouts & take best
    progress_every: int = 10
):
    actor_critic.eval()
    total_makespan, count = 0.0, 0

    # Flatten to per-instance
    def iter_instances():
        for batch in dataloader:
            T = batch["times"]
            M = batch["machines"]
            for i in range(len(T)):
                yield {"times": T[i], "machines": M[i]}

    def rollout_once(times, machines, stochastic: bool, precomputed_edge_index=None):
        env = JSSPEnvironment(times, machines, device=device, use_shaping_rewards=False)
        env.reset()
        
        max_steps = env.num_jobs * env.num_machines + 50
        steps = 0

        while True:
            steps += 1
            if steps > max_steps:
                return float("inf")  # guard

            available = env.get_available_actions()
            if not available:
                
                return float("inf")

            # Features/graph
            data = make_hetero_data(env, device, precomputed_edge_index=precomputed_edge_index)

            logits, _ = actor_critic(data)

            # Bool mask
            mask = torch.zeros(len(logits), dtype=torch.bool, device=device)
            mask[available] = True
            logits = logits.masked_fill(~mask, -1e10)

            if stochastic and best_of_k > 1:
                dist = Categorical(logits=logits)
                action = dist.sample().item()
            else:
                action = torch.argmax(logits).item()

            _, _, done, _ = env.step(action)
            if done:
                return env.get_makespan()

    for idx, inst in enumerate(iter_instances()):
        if idx >= limit_instances:
            break

        times = inst["times"]
        machines = inst["machines"]

        # Pre-calculate static graph topology once per instance
        from models.gin import HeteroGATv2
        static_edge_index = HeteroGATv2.build_edge_index_dict(machines)

        if best_of_k > 1:
            best = float("inf")
            for _ in range(best_of_k):
                ms = rollout_once(times, machines, stochastic=True, precomputed_edge_index=static_edge_index)
                if ms < best:
                    best = ms
            makespan = best
        else:
            makespan = rollout_once(times, machines, stochastic=False, precomputed_edge_index=static_edge_index)

        total_makespan += makespan
        count += 1

        if (idx + 1) % progress_every == 0:
            print(f"[VAL] {idx+1} instances -> avg {total_makespan / count:.2f}")

    avg_makespan = total_makespan / max(1, count)
    print(f"**** PPO Validation Avg Makespan (n={count}): {avg_makespan:.2f}")
    return avg_makespan
