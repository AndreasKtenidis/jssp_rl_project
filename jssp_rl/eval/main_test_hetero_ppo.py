"""main_test_ppo.py - Evaluate the best PPO/Reptile models on Taillard benchmark
===============================================================================
* Loads the checkpoint(s) saved (best_ppo.pt, meta_best.pth)
* Runs greedy policy (argmax) and optional best-of-K stochastic rollouts
* Saves makespans + gantt charts + CSV with both metrics.
"""

import os
import pickle
import torch
import pandas as pd
from models.gnn import GNNWithAttention
from models.actor_critic_ppo import ActorCriticPPO
from env.jssp_environment import JSSPEnvironment
from utils.logging_utils import plot_gantt_chart
from torch_geometric.data import Data
from utils.features import prepare_features

from config import BEST_OF_K,SAVE_GANTT_FOR_BEST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






def load_checkpoint(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No PPO checkpoint found at {path}. Train the model first.")
    state = torch.load(path, map_location=device)
    return state


@torch.no_grad()
def rollout_instance(env, model, stochastic=False):
    """Single rollout. If stochastic=True, sample from softmax instead of argmax."""
    done = False
    while not done:
        edge_index = GNNWithAttention.build_edge_index_with_machine_links(env.machines).to(device)
        x = prepare_features(env, edge_index, device)
        data = Data(x=x, edge_index=edge_index)

        logits, _ = model(data)

        avail = env.get_available_actions()
        mask = torch.zeros_like(logits)
        mask[avail] = 1
        logits = logits.masked_fill(mask == 0, -1e9)

        if stochastic:
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        else:
            action = torch.argmax(logits, dim=-1).item()

        _, _, done, _ = env.step(action)

    return env.get_makespan(), env.extract_job_assignments()


@torch.no_grad()
def run_rl_on_instance(env, model, best_of_k=1):
    """Run greedy once, and optionally best-of-K with stochastic sampling."""
    # Greedy run
    greedy_makespan, greedy_schedule = rollout_instance(env.clone(), model, stochastic=False)

    if best_of_k <= 1:
        return greedy_makespan, greedy_schedule, greedy_makespan, greedy_schedule

    # Best-of-K stochastic runs
    best_makespan = greedy_makespan
    best_schedule = greedy_schedule
    for _ in range(best_of_k):
        makespan_k, schedule_k = rollout_instance(env.clone(), model, stochastic=True)
        if makespan_k < best_makespan:
            best_makespan = makespan_k
            best_schedule = schedule_k

    return greedy_makespan, greedy_schedule, best_makespan, best_schedule




def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    saved_dir = os.path.join(base_dir, "saved")
    eval_dir = os.path.join(base_dir, "eval")
    model_dir = os.path.join(base_dir, "outputs", "models")

    os.makedirs(eval_dir, exist_ok=True)

    # Load Taillard benchmark instances
    taillard_pkl = os.path.join(saved_dir, "taillard_instances.pkl")
    with open(taillard_pkl, "rb") as f:
        instances = pickle.load(f)

    # Evaluate PPO and Reptile
    eval_configs = [
        {
            "name": "ppo",
            "model_path": os.path.join(model_dir, "best_ppo.pt"),
            "results_csv": os.path.join(eval_dir, "taillard_rl_results.csv"),
            "gantt_dir": os.path.join(eval_dir, "gantt_rl")
        },
        {
            "name": "reptile",
            "model_path": os.path.join(saved_dir, "meta_best.pth"),
            "results_csv": os.path.join(eval_dir, "taillard_reptile_results.csv"),
            "gantt_dir": os.path.join(eval_dir, "gantt_reptile")
        }
    ]

    for config in eval_configs:
        name = config["name"]
        model_path = config["model_path"]
        results_csv = config["results_csv"]
        gantt_dir = config["gantt_dir"]

        if not os.path.exists(model_path):
            print(f"⚠️ {name.upper()} model not found at {model_path}. Skipping...")
            continue

        os.makedirs(gantt_dir, exist_ok=True)

        # Build & load model
        model = ActorCriticPPO(
            node_input_dim=3,
            gnn_hidden_dim=128,
            gnn_output_dim=64,
            actor_hidden_dim=64,
            critic_hidden_dim=64,
            action_dim=15 * 15
        ).to(device)
        state_dict = load_checkpoint(model_path)
        model.load_state_dict(state_dict)
        model.eval()

        print(f"\n=== Evaluating {name.upper()} on Taillard ===")
        results = []
        for idx, inst in enumerate(instances):
            print(f" {name.upper()} testing on Taillard instance {idx + 1}/{len(instances)}")

            times = torch.tensor(inst["times"], dtype=torch.float32)
            machines = torch.tensor(inst["machines"], dtype=torch.long)
            env = JSSPEnvironment(times, machines, device=device)

            greedy_ms, greedy_sched, best_ms, best_sched = run_rl_on_instance(env, model, best_of_k=BEST_OF_K)

            results.append({
                "instance_id": idx,
                "rl_makespan_greedy": greedy_ms,
                f"rl_makespan_best_of_{BEST_OF_K}": best_ms
            })

            # Save gantt for the best schedule
            if SAVE_GANTT_FOR_BEST:
                gantt_path = os.path.join(gantt_dir, f"gantt_{name}_taillard_{idx:02d}.png")
                plot_gantt_chart(best_sched, save_path=gantt_path, show_op_index=True)

        pd.DataFrame(results).to_csv(results_csv, index=False)
        print(f"✅ {name.upper()} test results (greedy + best-of-K) saved to {results_csv}")


if __name__ == "__main__":
    main()
