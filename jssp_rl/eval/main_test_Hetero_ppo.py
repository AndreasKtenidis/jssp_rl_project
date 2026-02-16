# main_test_hetero_ppo.py
import os
import pickle
import torch
import pandas as pd

from models.Hetero_actor_critic_ppo import ActorCriticPPO
from env.jssp_environment import JSSPEnvironment
from utils.logging_utils import plot_gantt_chart
from utils.rollout_once import rollout_once

from config import BEST_OF_K, SAVE_GANTT_FOR_BEST




def load_checkpoint(path,device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No PPO checkpoint found at {path}. Train the model first.")
    return torch.load(path, map_location=device)


@torch.no_grad()
def evaluate_instance(times, machines, model, device,
                      *, best_of_k: int = 1,
                      use_est_boost=True, est_beta=1.0):
    """
    Επιστρέφει:
      - ms_stochastic, sched_stochastic: single-sample ή best-of-K sampling
      - ms_greedy,    sched_greedy: μία greedy κύλιση (argmax)
    """
    # (1) Stochastic: single ή best-of-K
    if best_of_k > 1:
        best = float("inf")
        best_sched = []
        for _ in range(best_of_k):
            env = JSSPEnvironment(times, machines)
            stats = rollout_once(env, model, device,
                                 mode="stochastic",
                                 use_est_boost=use_est_boost,
                                 est_beta=est_beta)
            if stats["makespan"] < best:
                best = stats["makespan"]
                best_sched = stats["env"].extract_job_assignments()
        ms_stochastic, sched_stochastic = best, best_sched
    else:
        env = JSSPEnvironment(times, machines)
        stats = rollout_once(env, model, device,
                             use_est_boost=use_est_boost,
                             est_beta=est_beta,
                             mode="stochastic",)
        ms_stochastic = stats["makespan"]
        sched_stochastic = stats["env"].extract_job_assignments()

    # (2) Greedy (argmax)
    env = JSSPEnvironment(times, machines)
    stats = rollout_once(env, model, device,
                         use_est_boost=use_est_boost,
                         est_beta=est_beta,
                         mode="greedy",)
    ms_greedy = stats["makespan"]
    sched_greedy = stats["env"].extract_job_assignments()

    return ms_stochastic, sched_stochastic, ms_greedy, sched_greedy


def main(device):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    saved_dir = os.path.join(base_dir, "saved")
    eval_dir = os.path.join(base_dir, "eval")
    model_path = os.path.join(base_dir, "outputs", "models", "best_ppo.pt")
    taillard_pkl = os.path.join(saved_dir, "taillard_instances.pkl")
    results_csv = os.path.join(eval_dir, "taillard_rl_results.csv")
    gantt_dir = os.path.join(eval_dir, "gantt_rl")

    os.makedirs(gantt_dir, exist_ok=True)

    # Load benchmark instances
    with open(taillard_pkl, "rb") as f:
        instances = pickle.load(f)

    # Build model & load checkpoint
    model = ActorCriticPPO(
        node_input_dim=13,
        gnn_hidden_dim=128,
        gnn_output_dim=64,
        actor_hidden_dim=64,
        critic_hidden_dim=64
    ).to(device)

    state_dict = load_checkpoint(model_path,device)
    model.load_state_dict(state_dict)
    model.eval()

    # Evaluate each instance
    rows = []
    sum_stoch, sum_greedy, n = 0.0, 0.0, 0

    for idx, inst in enumerate(instances):
        print(f" HeteroGIN PPO testing on Taillard instance {idx + 1}/{len(instances)}")

        times = torch.tensor(inst["times"], dtype=torch.float32)    # CPU
        machines = torch.tensor(inst["machines"], dtype=torch.long) # CPU

        ms_stoch, sched_stoch, ms_greedy, sched_greedy = evaluate_instance(
            times, machines, model, device,
            best_of_k=BEST_OF_K, use_est_boost=True, est_beta=1.0
        )

        delta = ms_greedy - ms_stoch
        print(f"  → stochastic(best-of-{BEST_OF_K}): {ms_stoch:.1f} | greedy: {ms_greedy:.1f} | Δ(greedy - stoch): {delta:.1f}")

        row = {
            "instance_id": idx,
            f"rl_makespan_stochastic_best_of_{BEST_OF_K}": ms_stoch,
            "rl_makespan_greedy": ms_greedy,
            "delta_greedy_minus_stochastic": delta
        }
        rows.append(row)

        # Gantt για το καλύτερο από τα δύο
        if SAVE_GANTT_FOR_BEST:
            if ms_stoch <= ms_greedy:
                best_sched, tag = sched_stoch, "stoch"
            else:
                best_sched, tag = sched_greedy, "greedy"
            gantt_path = os.path.join(gantt_dir, f"gantt_rl_taillard_{idx:02d}_{tag}.png")
            plot_gantt_chart(best_sched, save_path=gantt_path, show_op_index=True)

        sum_stoch += ms_stoch
        sum_greedy += ms_greedy
        n += 1

    # Αθροιστικά stats
    avg_stoch = sum_stoch / max(1, n)
    avg_greedy = sum_greedy / max(1, n)
    avg_delta = avg_greedy - avg_stoch
    print(f"\n==== Aggregate ====")
    print(f"Avg stochastic (best-of-{BEST_OF_K}): {avg_stoch:.2f}")
    print(f"Avg greedy:                       {avg_greedy:.2f}")
    print(f"Avg Δ(greedy - stoch):            {avg_delta:.2f}")

    pd.DataFrame(rows).to_csv(results_csv, index=False)
    print(f"✅ PPO (HeteroGIN) test results saved to {results_csv}")




if __name__ == "__main__":
    from main_train_Hetero_ppo import device
    main(device)
