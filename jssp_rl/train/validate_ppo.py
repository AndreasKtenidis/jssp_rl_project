# train/validate_ppo.py
import torch
from env.jssp_environment import JSSPEnvironment


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
            state = env.reset()
            done  = False

            while not done:
                # tensor‑ify state & pass through network
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).flatten().unsqueeze(0)
                logits, _ = actor_critic(state_tensor)

                # mask invalid actions
                avail = env.get_available_actions()
                mask  = torch.zeros_like(logits)
                mask[0, avail] = 1
                logits = logits.masked_fill(mask == 0, -1e10)

                action = torch.argmax(logits, dim=-1).item()   # Greedy
                state, _, done, _ = env.step(action)

            total_makespan += env.get_makespan()
            count += 1

    avg_makespan = total_makespan / count
    print(f"**** PPO Validation Avg Makespan: {avg_makespan:.2f}")
    return avg_makespan
