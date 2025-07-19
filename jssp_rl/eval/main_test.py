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

# Set device manually for testing mode
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_checkpoint(path):
    if not os.path.exists(path):
        print(f" No checkpoint found at {path}. Please train first.")
        exit(1)
    return torch.load(path, map_location=device)

def run_rl_on_instance(env, gnn, actor, edge_index, edge_weights):
    env.reset()
    done = False
    while not done:
        x = prepare_features(env, edge_index,device)
        node_embeddings = gnn(x, edge_index.to(device), edge_weights.to(device))
        _, logits = actor(node_embeddings, env)

        available_indices = env.get_available_actions()
        available_mask = torch.zeros_like(logits, dtype=torch.bool, device=device)
        available_mask[available_indices] = True

        masked_logits = logits.clone()
        masked_logits[~available_mask] = float('-inf')

        action = torch.argmax(masked_logits).item()
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
    gnn = GNNWithAttention(in_channels=3, hidden_dim=128, out_channels=64, num_heads=4).to(device)
    actor = Actor(gnn_output_dim=64, action_dim=15 * 15).to(device)

    # Load model
    ckpt = load_checkpoint(checkpoint_path)
    gnn.load_state_dict(ckpt["gnn_state_dict"])
    actor.load_state_dict(ckpt["actor_state_dict"])
    gnn.eval()
    actor.eval()

    results = []
    for i, inst in enumerate(instances):
        print(f"🧪 Testing RL on Taillard instance {i+1}/{len(instances)}")
        times = torch.tensor(inst["times"], dtype=torch.float32).to(device)
        machines = torch.tensor(inst["machines"], dtype=torch.long).to(device)

        env = JSSPEnvironment(times, machines, device=device)

        # Build edge_index with machine links
        edge_index = gnn.build_edge_index_with_machine_links(machines).to(device)
        edge_weights = torch.ones(edge_index.size(1), dtype=torch.float).to(device)

        makespan, schedule = run_rl_on_instance(env, gnn, actor, edge_index, edge_weights)

        results.append({"instance_id": i, "rl_makespan": makespan})
        gantt_path = os.path.join(gantt_dir, f"gantt_rl_taillard_{i:02d}.png")
        plot_gantt_chart(schedule, save_path=gantt_path, show_op_index=True)

    df = pd.DataFrame(results)
    df.to_csv(results_csv, index=False)
    print(f"✅ RL test results saved to {results_csv}")

if __name__ == "__main__":
    main()
