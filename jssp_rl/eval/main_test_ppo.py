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
from data.dataset import JSSPDataset
from models.Hetero_actor_critic_ppo import ActorCriticPPO
from env.jssp_environment import JSSPEnvironment
from utils.logging_utils import plot_gantt_chart
from torch_geometric.data import Data
from utils.features import make_hetero_data

from config import BEST_OF_K,SAVE_GANTT_FOR_BEST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






def load_checkpoint(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No PPO checkpoint found at {path}. Train the model first.")
    state = torch.load(path, map_location=device)
    return state


@torch.no_grad()
def rollout_instance_once(times, machines, model, stochastic=False):
    """Single rollout. If stochastic=True, sample from Categorical(logits)."""
    env = JSSPEnvironment(times, machines, device=device)
    env.reset()
    max_steps = env.num_jobs * env.num_machines + 50
    steps = 0

    while True:
        steps += 1
        if steps > max_steps:
            return float("inf"), []

        # edge_index = GNNWithAttention.build_edge_index_with_machine_links(env.machines).to(device)
        # x = prepare_features(env, edge_index, device)
        # data = Data(x=x, edge_index=edge_index)
        data = make_hetero_data(env, device)

        logits, _ = model(data)

        avail = env.get_available_actions()
        if not avail:
            return float("inf"), []

        mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=device)
        mask[avail] = True
        logits = logits.masked_fill(~mask, -1e10)

        if stochastic:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
        else:
            action = torch.argmax(logits).item()

        _, _, done, _ = env.step(action)
        if done:
            return env.get_makespan(), env.extract_job_assignments()


@torch.no_grad()
def run_rl_on_instance(times, machines, model, best_of_k=1):
    """Run greedy once, and optionally best-of-K with stochastic sampling."""
    greedy_ms, greedy_sched = rollout_instance_once(times, machines, model, stochastic=False)

    if best_of_k <= 1:
        return greedy_ms, greedy_sched, greedy_ms, greedy_sched

    best_makespan = greedy_ms
    best_schedule = greedy_sched
    for _ in range(best_of_k):
        makespan_k, schedule_k = rollout_instance_once(times, machines, model, stochastic=True)
        if makespan_k < best_makespan:
            best_makespan = makespan_k
            best_schedule = schedule_k

    return greedy_ms, greedy_sched, best_makespan, best_schedule





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

    # Automatically find all phase-specific checkpoints
    eval_configs = []
    
    # Standard checkpoints
    if os.path.exists(os.path.join(model_dir, "best_ppo_latest.pt")):
        eval_configs.append({
            "name": "ppo_latest",
            "model_path": os.path.join(model_dir, "best_ppo_latest.pt"),
            "results_csv": os.path.join(eval_dir, "taillard_ppo_latest.csv"),
            "gantt_dir": os.path.join(eval_dir, "gantt_ppo_latest")
        })

    # Phase-specific checkpoints
    for phase_name in ["Phase_1", "Phase_2", "Phase_3"]:
        ckpt = os.path.join(model_dir, f"best_ppo_{phase_name}.pt")
        if os.path.exists(ckpt):
            eval_configs.append({
                "name": f"ppo_{phase_name}",
                "model_path": ckpt,
                "results_csv": os.path.join(eval_dir, f"taillard_rl_{phase_name}.csv"),
                "gantt_dir": os.path.join(eval_dir, f"gantt_rl_{phase_name}")
            })
    
    # Reptile
    if os.path.exists(os.path.join(saved_dir, "meta_best_Phase_3.pth")):
         eval_configs.append({
            "name": "reptile_final",
            "model_path": os.path.join(saved_dir, "meta_best_Phase_3.pth"),
            "results_csv": os.path.join(eval_dir, "taillard_reptile_results.csv"),
            "gantt_dir": os.path.join(eval_dir, "gantt_reptile")
        })
    elif os.path.exists(os.path.join(saved_dir, "meta_best.pth")):
         eval_configs.append({
            "name": "reptile_legacy",
            "model_path": os.path.join(saved_dir, "meta_best.pth"),
            "results_csv": os.path.join(eval_dir, "taillard_reptile_results.csv"),
            "gantt_dir": os.path.join(eval_dir, "gantt_reptile")
        })


    for config in eval_configs:
        name = config["name"]
        model_path = config["model_path"]
        results_csv = config["results_csv"]
        gantt_dir = config["gantt_dir"]

        if not os.path.exists(model_path):
            print(f"[Warning] {name.upper()} model not found at {model_path}. Skipping...")
            continue

        os.makedirs(gantt_dir, exist_ok=True)

        # Build & load model
        # Build & load model
        model = ActorCriticPPO(
            node_input_dim=3,
            gnn_hidden_dim=64,
            gnn_output_dim=32,
            actor_hidden_dim=32,
            critic_hidden_dim=32,
            # action_dim=15 * 15  <-- removed for Hetero
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
            
            # Detect size
            N, M = times.shape

            greedy_ms, greedy_sched, best_ms, best_sched = run_rl_on_instance(times, machines, model, best_of_k=BEST_OF_K)

            results.append({
                "instance_id": idx,
                "size": f"{N}x{M}",
                "rl_makespan": greedy_ms,
                "rl_makespan_greedy": greedy_ms,
                f"rl_makespan_best_of_{BEST_OF_K}": best_ms
            })

            # Save gantt for the best schedule
            if SAVE_GANTT_FOR_BEST:
                gantt_path = os.path.join(gantt_dir, f"gantt_{name}_taillard_{idx:02d}.png")
                plot_gantt_chart(best_sched, save_path=gantt_path, show_op_index=True)

        pd.DataFrame(results).to_csv(results_csv, index=False)
        print(f"[Success] {name.upper()} test results (greedy + best-of-K) saved to {results_csv}")


if __name__ == "__main__":
    main()
