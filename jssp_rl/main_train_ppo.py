import os
import pickle
import subprocess
import pandas as pd
import torch
import argparse

from data.dataset import JSSPDataset, split_dataset, get_dataloaders
from env.jssp_environment import JSSPEnvironment
from models.actor_critic_ppo import ActorCriticPPO
from models.gnn import GNNWithAttention
from train.train_ppo import train
from train.validate_ppo import validate_ppo
from cp.main_cp import run_cp_on_taillard
from utils.logging_utils import (
    plot_rl_convergence,
    plot_gantt_chart,
    save_training_log_csv,
    plot_cp_vs_rl_comparison
)
from config import lr, num_epochs, batch_size

# # ── CLI ─────────────────────────────────────────────────────────────
# parser = argparse.ArgumentParser()
# parser.add_argument("--init", type=str, default=None,
#                     help="Path to a state-dict to warm-start the actor-critic")
# args = parser.parse_args()

device = torch.device("cpu")

# === Base Path Setup ===
base_dir = os.path.dirname(__file__)
outputs_dir = os.path.join(base_dir, "outputs")
log_dir = os.path.join(outputs_dir, "logs")
gantt_dir = os.path.join(outputs_dir, "gantt")
plot_dir = os.path.join(outputs_dir, "plots")
model_dir = os.path.join(outputs_dir, "models")
comparison_dir = os.path.join(outputs_dir, "comparisons")

# Create all directories
for d in [log_dir, gantt_dir, plot_dir, model_dir, comparison_dir]:
    os.makedirs(d, exist_ok=True)

# === Load Dataset ===
with open(os.path.join(base_dir, "saved", "synthetic_500_instances.pkl"), "rb") as f:
    instances = pickle.load(f)

# === Dataset Setup ===
dataset = JSSPDataset(instances)
split_dataset(dataset)
dataloaders = get_dataloaders(dataset, batch_size=batch_size)

# === Model Setup ===
gnn = GNNWithAttention(in_channels=3, hidden_dim=128, out_channels=64, num_heads=4).to(device)
actor_critic = ActorCriticPPO(
    node_input_dim=3,
    gnn_hidden_dim=128,
    gnn_output_dim=64,
    actor_hidden_dim=64,
    critic_hidden_dim=64,
    action_dim=15 * 15
).to(device)


# # === Warm-start from Reptile θ★  ===================================
# if args.init is not None:
#     print(f"🔄 Loading warm-start weights from {args.init}")
#     state = torch.load(args.init, map_location=device)
#     actor_critic.load_state_dict(state, strict=True)
# else:
#     print("⚠️  No warm-start supplied: training will start from scratch")

optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

# === Training ===
best_val_makespan = float("inf")
all_makespans = []
log_data = []

actor_critic.train()
for epoch in range(num_epochs):
    print(f"\n=== PPO Epoch {epoch+1}/{num_epochs} ===")

    train_makespan, loss = train(
        dataloaders['train'], actor_critic, optimizer, device=device
    )

    actor_critic.eval()
    val_makespan = validate_ppo(dataloaders['val'], actor_critic, device=device)
    actor_critic.train()

    all_makespans.append(train_makespan)
    log_data.append({
        "epoch": epoch + 1,
        "train_loss": loss,
        "train_best_makespan": train_makespan,
        "val_makespan": val_makespan
    })

    save_training_log_csv(log_data, filename=os.path.join(log_dir, "training_log_ppo.csv"))

    if val_makespan < best_val_makespan:
        best_val_makespan = val_makespan
        torch.save(actor_critic.state_dict(), os.path.join(model_dir, "best_ppo.pt"))
        print("Saved new best PPO model.")

# === Plot Convergence ===
plot_rl_convergence(all_makespans, save_path=os.path.join(plot_dir, "rl_convergence.png"))

# === Gantt Chart for First Instance ===
example = instances[0]
actor_critic.eval()
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

plot_gantt_chart(env.extract_job_assignments(), save_path=os.path.join(gantt_dir, "gantt_chart.png"))

# === Run CP on Taillard ===
print("\nRunning CP on Taillard for comparison...")
run_cp_on_taillard()

# === Run RL on Taillard ===
print("\nRunning RL on Taillard benchmark instances...")
subprocess.call(["python", "-m", "eval.main_test_ppo"], cwd=base_dir)

# === Merge and Compare RL vs CP ===
cp_csv = os.path.join(base_dir, "cp", "cp_makespans.csv")
rl_csv = os.path.join(base_dir, "eval", "taillard_rl_results.csv")
merged_csv = os.path.join(comparison_dir, "taillard_comparison.csv")
plot_path = os.path.join(plot_dir, "cp_vs_rl_barplot.png")

cp_df = pd.read_csv(cp_csv)
rl_df = pd.read_csv(rl_csv)
merged = pd.merge(cp_df, rl_df, on="instance_id")
merged["gap"] = (merged["rl_makespan"] - merged["cp_makespan"]) / merged["cp_makespan"]
merged.to_csv(merged_csv, index=False)

print(f"Saved merged CP vs RL comparison to {merged_csv}")
plot_cp_vs_rl_comparison(merged, save_path=plot_path)
