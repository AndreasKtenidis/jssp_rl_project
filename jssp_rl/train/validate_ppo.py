# train/validate_ppo.py  (GNN / homogeneous version)

import torch
from torch.distributions import Categorical
from torch_geometric.data import Data
from env.jssp_environment import JSSPEnvironment
from models.gnn import GNNWithAttention
from utils.features import prepare_features
from utils.action_masking import select_top_k_actions

@torch.no_grad()
def validate_ppo(
    dataloader,
    actor_critic,
    device,
    *,
    limit_instances: int = 50,   # cap validation size
    best_of_k: int = 1,          # >1 to try sampling K times and take best
    progress_every: int = 10     # print progress every N instances
):
    actor_critic.eval()
    total_makespan, count = 0.0, 0

    # Flatten dataloader into single instances
    def iter_instances():
        for batch in dataloader:
            T = batch["times"]
            M = batch["machines"]
            for i in range(len(T)):
                yield {"times": T[i], "machines": M[i]}

    for idx, inst in enumerate(iter_instances()):
        if idx >= limit_instances:
            break

        times = inst["times"]
        machines = inst["machines"]
        env = JSSPEnvironment(times, machines, device=device)
        num_jobs, num_machines = env.num_jobs, env.num_machines
        max_steps = num_jobs * num_machines + 50  # hard cap

        def rollout(sample: bool = False):
            env.reset()
            steps = 0
            while True:
                if steps > max_steps:
                    return float("inf")   # guard against infinite loops
                steps += 1

                available = env.get_available_actions()
                if not available:
                    # Fallback: heuristic pick; if still none, bail
                    topk, _ = select_top_k_actions(env, top_k=1, device=device)
                    if topk.numel() == 0:
                        return float("inf")
                    action = int(topk[0].item())
                    _, _, done, _ = env.step(action)
                    if done:
                        return env.get_makespan()
                    continue

                # Build features/graph (GNN version)
                edge_index = GNNWithAttention.build_edge_index_with_machine_links(env.machines).to(device)
                x = prepare_features(env, edge_index, device)  # [num_ops, feat]
                data = Data(x=x, edge_index=edge_index)

                logits, _ = actor_critic(data)

                mask = torch.zeros(len(logits), dtype=torch.bool, device=device)
                mask[available] = True
                logits = logits.masked_fill(~mask, -1e10)

                if sample and best_of_k > 1:
                    dist = Categorical(logits=logits)
                    action = dist.sample().item()
                else:
                    action = torch.argmax(logits).item()

                _, _, done, _ = env.step(action)
                if done:
                    return env.get_makespanspan()  # or env.get_makespan() if that's your method name

        if best_of_k > 1:
            best = float("inf")
            for _ in range(best_of_k):
                ms = rollout(sample=True)
                if ms < best:
                    best = ms
            makespan = best
        else:
            makespan = rollout(sample=False)

        total_makespan += makespan
        count += 1

        if (idx + 1) % progress_every == 0:
            print(f"[VAL] {idx+1} instances → avg {total_makespan / count:.2f}")

    avg_makespan = total_makespan / max(1, count)
    print(f"**** PPO Validation Avg Makespan (n={count}): {avg_makespan:.2f}")
    return avg_makespan
