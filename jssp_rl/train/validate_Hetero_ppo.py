import torch
from env.jssp_environment import JSSPEnvironment
from utils.Heterofeatures import prepare_features

@torch.no_grad()
def validate_ppo(
    dataloader,
    actor_critic,
    device,
    *,
    limit_instances: int = 50,   # cap validation size
    best_of_k: int = 10,          # >1 → K stochastic rollouts & keep best
    progress_every: int = 10,
    report_greedy: bool = True,  # βγάλε και greedy metric
    return_both: bool = True     # αν True, επιστρέφει (avg_stochastic, avg_greedy)
):
    """
    Validation συμβατό με το baseline train():
    - Χωρίς heuristics/fallbacks
    - Categorical sampling για stochastic, argmax για greedy
    - Bool legal mask
    - best_of_k: τρέχεις K στοχαστικά rollouts και κρατάς min makespan
    """
    actor_critic.eval()

    # flatten per-instance από τον dataloader
    def iter_instances():
        for batch in dataloader:
            T = batch["times"]
            M = batch["machines"]
            for i in range(len(T)):
                yield {"times": T[i], "machines": M[i]}

    def rollout_once(times, machines, *, stochastic: bool):
        # ίδιο init με train()
        env = JSSPEnvironment(times, machines)
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

            # Features στο σωστό device (όπως στην train)
            data = prepare_features(env, device)

            logits, _ = actor_critic(data)   # [num_ops]
            logits = logits.view(-1)

            # Bool legal mask
            mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
            mask[torch.as_tensor(available, device=logits.device, dtype=torch.long)] = True
            masked_logits = logits.masked_fill(~mask, -1e10)

            if stochastic:
                dist = torch.distributions.Categorical(logits=masked_logits)
                action = int(dist.sample().item())
            else:
                action = int(masked_logits.argmax().item())

            _, _, done, _ = env.step(action)
            if done:
                return env.get_makespan()

    # συσσωρευτές
    total_stoch, total_greedy, n = 0.0, 0.0, 0

    for idx, inst in enumerate(iter_instances()):
        if idx >= limit_instances:
            break

        times = inst["times"]
        machines = inst["machines"]

        # Stochastic metric: single ή best-of-K
        if best_of_k > 1:
            best = float("inf")
            for _ in range(best_of_k):
                ms = rollout_once(times, machines, stochastic=True)
                if ms < best:
                    best = ms
            ms_stoch = best
        else:
            ms_stoch = rollout_once(times, machines, stochastic=True)

        # Greedy metric (προαιρετικό)
        if report_greedy:
            ms_greedy = rollout_once(times, machines, stochastic=False)
            delta = ms_greedy - ms_stoch
            print(f"[VAL {idx+1}] stoch(best-of-{best_of_k})={ms_stoch:.1f} | greedy={ms_greedy:.1f} | Δ={delta:.1f}")
            total_greedy += ms_greedy
        else:
            print(f"[VAL {idx+1}] stoch(best-of-{best_of_k})={ms_stoch:.1f}")

        total_stoch += ms_stoch
        n += 1

        if (idx + 1) % progress_every == 0:
            if report_greedy:
                print(f"[VAL] {idx+1} inst → avg stoch={total_stoch/n:.2f} | avg greedy={total_greedy/n:.2f}")
            else:
                print(f"[VAL] {idx+1} inst → avg stoch={total_stoch/n:.2f}")

    avg_stoch = total_stoch / max(1, n)
    if report_greedy:
        avg_greedy = total_greedy / max(1, n)
        print(f"**** PPO Validation Avg Makespan (n={n}) → stoch(best-of-{best_of_k})={avg_stoch:.2f} | greedy={avg_greedy:.2f}")
        return (avg_stoch, avg_greedy) if return_both else avg_stoch
    else:
        print(f"**** PPO Validation Avg Makespan (n={n}) → stoch(best-of-{best_of_k})={avg_stoch:.2f}")
        return avg_stoch
