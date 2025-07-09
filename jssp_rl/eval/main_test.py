# jssp_rl/eval/main_test.py

import os
import pickle
import torch
import pandas as pd
from models.gnn import GNNWithAttention
from models.actor_critic import Actor
from env.jssp_environment import JSSPEnvironment
from utils.features import prepare_features
from utils.logging_utils import plot_gantt_chart
from utils.action_masking import select_top_k_actions

def load_checkpoint(path):
    if not os.path.exists(path):
        print(f" No checkpoint found at {path}. Please train first.")
        exit(1)
    return torch.load(path)

def run_rl_on_instance(env, gnn, actor, edge_index, edge_weights):
    env.reset()
    done = False
    while not done:
        x = prepare_features(env, edge_index)
        node_embeddings = gnn(x, edge_index, edge_weights)
        logits = actor(node_embeddings, env)
        available = env.get_available_actions()
        action = select_top_k_actions(logits, available, k=1)[0]
        _, _, done, _ = env.step(action)
    return env.get_makespan(), env.extract_job_assignments()

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    taillard_path = os.path.join(base_dir, "saved", "taillard_instances.pkl")
    checkpoint_path = os.path.join("/home/aktenidis/JSSPprojects/jssp_rl_project", "checkpoints", "best_model.pth")
    results_csv = os.path.join(base_dir, "eval", "taillard_rl_results.csv")
    gantt_dir = os.path.join(base_dir, "eval", "gantt_rl")

    with open(taillard_path, "rb") as f:
        instances = pickle.load(f)

    os.makedirs(gantt_dir, exist_ok=True)

    # Model definition
    gnn = GNNWithAttention(in_channels=3, hidden_dim=128, out_channels=64, num_heads=4)
    actor = Actor(gnn_output_dim=64, action_dim=15 * 15)

    # Create dummy edge_index
    num_jobs, num_machines = instances[0]['times'].shape
    edge_list = []
    for j in range(num_jobs):
        for o in range(num_machines - 1):
            u = j * num_machines + o
            v = j * num_machines + o + 1
            edge_list.append([u, v])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weights = torch.ones(edge_index.size(1), dtype=torch.float)

    # Load model
    ckpt = load_checkpoint(checkpoint_path)
    gnn.load_state_dict(ckpt["gnn_state_dict"])
    actor.load_state_dict(ckpt["actor_state_dict"])
    gnn.eval()
    actor.eval()

    results = []
    for i, inst in enumerate(instances):
        print(f"🧪 Testing RL on Taillard instance {i+1}/{len(instances)}")
        env = JSSPEnvironment(inst["times"], inst["machines"])
        makespan, schedule = run_rl_on_instance(env, gnn, actor, edge_index, edge_weights)

        results.append({"instance_id": i, "rl_makespan": makespan})
        gantt_path = os.path.join(gantt_dir, f"gantt_rl_taillard_{i:02d}.html")
        plot_gantt_chart(schedule, save_path=gantt_path)

    df = pd.DataFrame(results)
    df.to_csv(results_csv, index=False)
    print(f"✅ RL test results saved to {results_csv}")

if __name__ == "__main__":
    main()
