import torch
from env.jssp_environment import JSSPEnvironment
from utils.rollout_once import rollout_once


@torch.no_grad()
def validate_ppo(
    dataloader,
    actor_critic,
    device,
    use_est_boost,
    est_beta,
    *,
    limit_instances: int = 50,   # cap validation size
    best_of_k: int = 10,         # stochastic rollouts & keep best
    progress_every: int = 10,
    report_greedy: bool = True,  # print greedy metric
    return_both: bool = True,    # return (avg_stoch, avg_greedy)
    
):
    """
    Validation:
      - Greedy rollout (argmax, single run)
      - Best-of-K stochastic rollouts (sampled, keep best)
    """

    # flatten per-instance from dataloader
    def iter_instances():
        for batch in dataloader:
            T = batch["times"]
            M = batch["machines"]
            for i in range(len(T)):
                yield {"times": T[i], "machines": M[i]}

    total_stoch, total_greedy, n = 0.0, 0.0, 0

    for idx, inst in enumerate(iter_instances()):
        if idx >= limit_instances:
            break

        env = JSSPEnvironment(inst["times"], inst["machines"])

        # (1) Greedy rollout (argmax)
        stats_greedy = rollout_once(
            env,
            actor_critic,
            device,
            use_est_boost=use_est_boost,
            est_beta=est_beta,
            mode="greedy",   # always eval
        )
        ms_greedy = stats_greedy["makespan"]

        # (2) stochastic best-of-K (sample)
        best_stoch = float("inf")
        for _ in range(best_of_k):
            env_k = JSSPEnvironment(inst["times"], inst["machines"])
            _, stats_stoch = rollout_once(
                env_k,
                actor_critic,
                device,
                use_est_boost=use_est_boost,
                est_beta=est_beta,
                mode="stochastic",   # sample actions
            )
            best_stoch = min(best_stoch, stats_stoch["makespan"])
        ms_stoch = best_stoch

        # --- print per-instance ---
        delta = ms_greedy - ms_stoch
        if report_greedy:
            print(
                f"[VAL {idx+1}] greedy={ms_greedy:.1f} | "
                f"best-of-{best_of_k} stochastic={ms_stoch:.1f} | Δ={delta:.1f}"
            )
        else:
            print(f"[VAL {idx+1}] best-of-{best_of_k} stochastic={ms_stoch:.1f}")

        # aggregate
        total_stoch += ms_stoch
        total_greedy += ms_greedy
        n += 1

        # --- progress logging ---
        if (idx + 1) % progress_every == 0:
            if report_greedy:
                print(
                    f"[VAL] {idx+1} inst → "
                    f"avg greedy={total_greedy/n:.2f} | "
                    f"avg best-of-{best_of_k} stoch={total_stoch/n:.2f}"
                )
            else:
                print(f"[VAL] {idx+1} inst → avg stoch={total_stoch/n:.2f}")

    # --- final averages ---
    avg_stoch = total_stoch / max(1, n)
    avg_greedy = total_greedy / max(1, n)

    if report_greedy:
        print(
            f"**** PPO Validation Avg Makespan (n={n}) → "
            f"greedy={avg_greedy:.2f} | best-of-{best_of_k} stoch={avg_stoch:.2f}"
        )
        return (avg_stoch, avg_greedy) if return_both else avg_stoch
    else:
        print(
            f"**** PPO Validation Avg Makespan (n={n}) → "
            f"best-of-{best_of_k} stoch={avg_stoch:.2f}"
        )
        return avg_stoch
