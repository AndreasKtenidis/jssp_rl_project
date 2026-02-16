# jssp_rl/main_train.py

import pickle
import torch
import subprocess
import pandas as pd
import os
from data.dataset import JSSPDataset, split_dataset, get_dataloaders
from models.gnn import GNNWithAttention
from jssp_rl_project.jssp_rl.models.actor_critic_a2c import Actor, Critic
from env.jssp_environment import JSSPEnvironment
from jssp_rl_project.jssp_rl.train.train_loop_a2c import train_loop
from train.validate import validate
from cp.main_cp import run_cp_on_taillard
from utils.logging_utils import (
    plot_rl_convergence,
    plot_gantt_chart,
    save_training_log_csv,
    plot_cp_vs_rl_comparison
)

# === Device Setup ===
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("CUDA available:", torch.cuda.is_available())
# print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# === Base path ===
base_dir = os.path.dirname(__file__)

# === Load Data ===
with open(os.path.join(base_dir, "saved", "synthetic_500_instances.pkl"), "rb") as f:
    instances = pickle.load(f)

# === Dataset and Dataloader ===
dataset = JSSPDataset(instances)
split_dataset(dataset)
dataloaders = get_dataloaders(dataset, batch_size=16)

# === Create Models and Move to Device ===
gnn = GNNWithAttention(in_channels=3, hidden_dim=128, out_channels=64, num_heads=4).to(device)
actor = Actor(gnn_output_dim=64, action_dim=15 * 15).to(device)
critic = Critic(gnn_output_dim=64).to(device)

# === Edge Index with Job + Machine links ===
machines_tensor = instances[0]['machines'].long()
edge_index = gnn.build_edge_index_with_machine_links(machines_tensor).to(device)
edge_weights = torch.ones(edge_index.size(1), dtype=torch.float).to(device)

# === Optimizer ===
optimizer = torch.optim.Adam(
    list(gnn.parameters()) + list(actor.parameters()) + list(critic.parameters()),
    lr=1e-3
)

# === Run Training with Validation ===
log_data = []
num_epochs = 30
all_makespans = []
all_losses = []

for epoch in range(num_epochs):
    print(f"\n Epoch {epoch+1}/{num_epochs}")

    episode_makespans, losses = train_loop(
        dataloader=dataloaders['train'],
        gnn=gnn,
        actor=actor,
        critic=critic,
        edge_index=edge_index,
        edge_weights=edge_weights,
        optimizer=optimizer,
        epochs=1,
        validate_fn=validate,
        val_dataloader=dataloaders['val'],
        device=device  
    )

    if episode_makespans and losses:
        val_makespan = validate(
            dataloaders['val'],
            gnn, actor, critic,
            edge_index, edge_weights,
            device=device
        )

        log_data.append({
            "epoch": epoch + 1,
            "train_loss": losses[-1],
            "train_best_makespan": episode_makespans[-1],
            "val_makespans": val_makespan
        })

        all_makespans.append(episode_makespans[-1])
        all_losses.append(losses[-1])
        save_training_log_csv(log_data, filename=os.path.join(base_dir, "training_log.csv"))
    else:
        print(" No makespan or loss returned from train_loop. Skipping log.")

# === Plot Convergence ===
plot_rl_convergence(all_makespans, save_path=os.path.join(base_dir, "rl_convergence.png"))

# === Plot Gantt Chart (replay one instance for visualization)
env = JSSPEnvironment(instances[0]['times'], instances[0]['machines'])
env.reset()

done = False
while not done:
    x = env.extract_job_assignments()
    for job in range(env.num_jobs):
        for op in range(env.num_machines):
            if env.state[job, op] == 0 and (op == 0 or env.state[job, op - 1] == 1):
                idx = job * env.num_machines + op
                _, _, done, _ = env.step(idx)
                break
        if done:
            break

plot_gantt_chart(env.extract_job_assignments(), save_path=os.path.join(base_dir, "gantt_chart.png"))

# === Run CP on Taillard
print("\n Running CP on Taillard for comparison...")
run_cp_on_taillard()

# === Run RL on Taillard
print("\n Running RL on Taillard benchmark instances...")
subprocess.call(["python", "-m", "eval.main_test"], cwd=base_dir)

# === Merge and compare RL vs CP ===
cp_csv = os.path.join(base_dir, "cp", "cp_makespans.csv")
rl_csv = os.path.join(base_dir, "eval", "taillard_rl_results.csv")
merged_csv = os.path.join(base_dir, "eval", "taillard_comparison.csv")
plot_path = os.path.join(base_dir, "eval", "cp_vs_rl_barplot.png")

cp_df = pd.read_csv(cp_csv)
rl_df = pd.read_csv(rl_csv)
merged = pd.merge(cp_df, rl_df, on="instance_id")
merged["gap"] = (merged["rl_makespan"] - merged["cp_makespan"]) / merged["cp_makespan"]
merged.to_csv(merged_csv, index=False)

print(f" Saved merged CP vs RL comparison to {merged_csv}")
plot_cp_vs_rl_comparison(merged, save_path=plot_path)
