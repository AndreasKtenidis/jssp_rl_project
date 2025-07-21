import os
import pickle
import subprocess
import pandas as pd
import torch

from data.dataset import JSSPDataset, split_dataset, get_dataloaders
from env.jssp_environment import JSSPEnvironment
from models.actor_critic_ppo import ActorCriticPPO
from models.gnn import GNNWithAttention
from train.train_ppo import train_ppo
from train.validate_ppo import validate_ppo
from cp.main_cp import run_cp_on_taillard
from utils.logging_utils import (
    plot_rl_convergence,
    plot_gantt_chart,
    save_training_log_csv,
    plot_cp_vs_rl_comparison
)
from config import lr, num_epochs, batch_size

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Base Path ===
base_dir = os.path.dirname(__file__)

# === Load Synthetic Dataset ===
with open(os.path.join(base_dir, "saved", "synthetic_500_instances.pkl"), "rb") as f:
    instances = pickle.load(f)

# === Dataset Setup ===
dataset = JSSPDataset(instances)
split_dataset(dataset)
dataloaders = get_dataloaders(dataset, batch_size=batch_size)

# === Model Setup ===
gnn = GNNWithAttention(in_channels=3, hidden_dim=128, out_channels=64, num_heads=4).to(device)
actor_critic = ActorCriticPPO(gnn, action_dim=15*15).to(device)
optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

# === Training ===
best_val_makespan = float("inf")
all_makespans = []
log_data = []

for epoch in range(num_epochs):
    print(f"\n=== PPO Epoch {epoch+1}/{num_epochs} ===")

    train_makespan, loss = train_ppo(
        dataloaders['train'], actor_critic, optimizer, device=device
    )

    val_makespan = validate_ppo(dataloaders['val'], actor_critic, device=device)
    all_makespans.append(train_makespan)

    log_data.append({
        "epoch": epoch + 1,
        "train_loss": loss,
        "train_best_makespan": train_makespan,
        "val_makespan": val_makespan
    })

    save_training_log_csv(log_data, filename=os.path.join(base_dir, "training_log_ppo.csv"))

    # Save best model
    if val_makespan < best_val_makespan:
        best_val_makespan = val_makespan
        torch.save(actor_critic.state_dict(), os.path.join(base_dir, "best_ppo.pt"))
        print("Saved new best PPO model.")

# === Plot Convergence ===
plot_rl_convergence(all_makespans, save_path=os.path.join(base_dir, "rl_convergence.png"))

# === Gantt Chart for first instance ===
example = instances[0]
env = JSSPEnvironment(example['times'], example['machines'])
done = False
env.reset()
while not done:
    available = env.get_available_actions()
    state = torch.tensor(env.state, dtype=torch.float32, device=device).flatten().unsqueeze(0)
    logits, _ = actor_critic(state)
    mask = torch.zeros_like(logits)
    mask[0, available] = 1
    logits = logits.masked_fill(mask == 0, -1e10)
    action = torch.argmax(logits, dim=-1).item()
    _, _, done, _ = env.step(action)

plot_gantt_chart(env.extract_job_assignments(), save_path=os.path.join(base_dir, "gantt_chart.png"))

# === Run CP on Taillard ===
print("\nRunning CP on Taillard for comparison...")
run_cp_on_taillard()

# === Run RL on Taillard ===
print("\nRunning RL on Taillard benchmark instances...")
subprocess.call(["python", "-m", "eval.main_test_ppo"], cwd=base_dir)

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

print(f"Saved merged CP vs RL comparison to {merged_csv}")
plot_cp_vs_rl_comparison(merged, save_path=plot_path)
