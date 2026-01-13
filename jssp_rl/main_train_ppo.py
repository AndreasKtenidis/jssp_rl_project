import os
import sys
import pickle
import subprocess
import pandas as pd
import torch

from data.dataset import JSSPDataset, split_dataset, get_dataloaders
from env.jssp_environment import JSSPEnvironment
from models.actor_critic_ppo import ActorCriticPPO
from models.gnn import GNNWithAttention
from reptile.meta_reptile import reptile_meta_train  

from train.train_ppo import train
from train.validate_ppo import validate_ppo
from cp.main_cp import run_cp_on_taillard
from utils.features import prepare_features
from torch_geometric.data import Data
from utils.logging_utils import (
    plot_rl_convergence,
    plot_gantt_chart,
    save_training_log_csv,
    plot_cp_vs_rl_comparison
) 
from config import lr, num_epochs, batch_size,VAL_LIMIT,BEST_OF_K,LOG_BOTH,PROGRESS_EVERY

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
with open(os.path.join(base_dir, "saved", "Synthetic_instances_15x15_5000.pkl"), "rb") as f:
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

# === Run Reptile Meta-Learning ===
meta_ckpt_path = os.path.join(base_dir, "saved", "meta_best.pth")

print("\n🔁 [Step 1] Running Reptile Meta-Training...")
actor_critic = reptile_meta_train(
    task_loader=dataloaders['train'],
    actor_critic=actor_critic,
    val_loader=dataloaders['val'],
    meta_iterations=300,          
    meta_batch_size=16,           
    inner_steps=3,               
    inner_lr=2e-4,
    meta_lr=2e-3,
    device=device,
    save_path=meta_ckpt_path,
    
    inner_update_batch_size_size=4,
    inner_switch_epoch=1,
    validate_every=10,            
)


print(" Saved best θ★ from Reptile at:", meta_ckpt_path)

# === Load θ★ from Reptile for PPO Warm Start ===
actor_critic.load_state_dict(torch.load(meta_ckpt_path, map_location=device))
print(f"✅ Loaded warm-start θ★ for PPO training from: {meta_ckpt_path}")

optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)


# === Training ===
best_val_makespan = float("inf")
all_makespans = []
log_data = []

actor_critic.train()
for epoch in range(num_epochs):
    print(f"\n=== PPO Epoch {epoch+1}/{num_epochs} ===")

    train_dataset = dataset.get_split("train")  
    train_makespan, loss = train(
        train_dataset, actor_critic, optimizer, device=device, update_batch_size_size=32
    )

    actor_critic.eval()
    # Greedy eval (deterministic argmax)
    val_makespan_greedy = validate_ppo(
        dataloader=dataloaders['val'],
        actor_critic=actor_critic,
        device=device,
        limit_instances=VAL_LIMIT,
        best_of_k=1,
        progress_every=PROGRESS_EVERY,
    )

    # Best-of-K eval (optional)
    if LOG_BOTH and BEST_OF_K > 1:
        val_makespan_best = validate_ppo(
            dataloader=dataloaders['val'],
            actor_critic=actor_critic,
            device=device,
            limit_instances=VAL_LIMIT,
            best_of_k=BEST_OF_K,
            progress_every=PROGRESS_EVERY,
        )
    else:
        val_makespan_best = val_makespan_greedy
    actor_critic.train()

    all_makespans.append(train_makespan)
    log_data.append({
    "epoch": epoch + 1,
    "train_loss": loss,
    "train_best_makespan": train_makespan,
    "val_makespan_greedy": val_makespan_greedy,
    "val_makespan_best_of_k": val_makespan_best,
    "best_of_k": BEST_OF_K,
    "val_limit": VAL_LIMIT,
})

    save_training_log_csv(log_data, filename=os.path.join(log_dir, "training_log_ppo.csv"))

    score_for_saving = val_makespan_best  # or val_makespan_greedy
if score_for_saving < best_val_makespan:
    best_val_makespan = score_for_saving
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
    edge_index = GNNWithAttention.build_edge_index_with_machine_links(env.machines)
    x = prepare_features(env, edge_index, device)
    data = Data(x=x, edge_index=edge_index.to(device))
    logits, _ = actor_critic(data)

    mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=device)
    mask[available] = True
    logits = logits.masked_fill(~mask, -1e10)

    action = torch.argmax(logits, dim=-1).item()
    _, _, done, _ = env.step(action)

plot_gantt_chart(env.extract_job_assignments(), save_path=os.path.join(gantt_dir, "gantt_chart.png"))

# === Run CP on Taillard ===
print("\nRunning CP on Taillard for comparison...")
run_cp_on_taillard()

# === Run RL on Taillard ===
print("\nRunning RL on Taillard benchmark instances...")
subprocess.call([sys.executable, "-m", "eval.main_test_ppo"], cwd=base_dir)


# === Merge and Compare CP vs PPO vs Reptile ===
cp_csv = os.path.join(base_dir, "cp", "cp_makespans.csv")
ppo_csv = os.path.join(base_dir, "eval", "taillard_rl_results.csv")
reptile_csv = os.path.join(base_dir, "eval", "taillard_reptile_results.csv")
merged_csv = os.path.join(comparison_dir, "taillard_comparison.csv")
plot_path = os.path.join(plot_dir, "cp_vs_rl_barplot.png")

cp_df = pd.read_csv(cp_csv)
ppo_df = pd.read_csv(ppo_csv)
merged = pd.merge(cp_df, ppo_df, on="instance_id")

if os.path.exists(reptile_csv):
    reptile_df = pd.read_csv(reptile_csv)
    merged = pd.merge(merged, reptile_df, on="instance_id", suffixes=("", "_reptile"))
    merged["gap_reptile"] = (merged["rl_makespan_reptile"] - merged["cp_makespan"]) / merged["cp_makespan"]

merged["gap"] = (merged["rl_makespan"] - merged["cp_makespan"]) / merged["cp_makespan"]
merged.to_csv(merged_csv, index=False)

print(f"Saved merged CP vs PPO vs Reptile comparison to {merged_csv}")
plot_cp_vs_rl_comparison(merged, save_path=plot_path)

print("\n Full pipeline complete: Reptile → PPO → CP vs RL evaluation.")

