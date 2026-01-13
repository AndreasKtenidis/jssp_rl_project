import os
import sys
import pickle
import subprocess
import pandas as pd
import torch

from data.dataset import JSSPDataset, split_dataset, get_dataloaders
from env.jssp_environment import JSSPEnvironment
from models.Hetero_actor_critic_ppo import ActorCriticPPO
from reptile.meta_reptile_Hetero import reptile_meta_train

from train.train_Hetero_ppo import train
from train.validate_Hetero_ppo import validate_ppo
from cp.main_cp import run_cp_on_taillard
from utils.Heterofeatures import prepare_features
from utils.logging_utils import (
    plot_rl_convergence,
    plot_gantt_chart,
    save_training_log_csv,
    plot_cp_vs_rl_comparison,
)

from config import lr, num_epochs, batch_size, VAL_LIMIT, BEST_OF_K, PROGRESS_EVERY


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Base Path Setup ===
base_dir = os.path.dirname(__file__)
outputs_dir = os.path.join(base_dir, "outputs")
log_dir = os.path.join(outputs_dir, "logs")
gantt_dir = os.path.join(outputs_dir, "gantt")
plot_dir = os.path.join(outputs_dir, "plots")
model_dir = os.path.join(outputs_dir, "models")
comparison_dir = os.path.join(outputs_dir, "comparisons")

for d in [log_dir, gantt_dir, plot_dir, model_dir, comparison_dir]:
    os.makedirs(d, exist_ok=True)

# === Load Dataset ===
with open(os.path.join(base_dir, "saved", "Synthetic_instances_15x15_505.pkl"), "rb") as f:
    instances = pickle.load(f)

dataset = JSSPDataset(instances)
split_dataset(dataset)
dataloaders = get_dataloaders(dataset, batch_size=batch_size)

# === Model ===
actor_critic = ActorCriticPPO(
    node_input_dim=13,
    gnn_hidden_dim=128,
    gnn_output_dim=64,
    actor_hidden_dim=64,
    critic_hidden_dim=64,
).to(device)

# === Reptile Meta-Learning ===
meta_ckpt_path = os.path.join(base_dir, "saved", "meta_best.pth")
print("\n🔁 [Step 1] Running Reptile Meta-Training...")
actor_critic = reptile_meta_train(
    task_loader=dataloaders['train'],
    actor_critic=actor_critic,
    val_loader=dataloaders['val'],
    meta_iterations=100,
    meta_batch_size=16,
    inner_steps=1,
    inner_lr=1e-4,
    meta_lr=1e-3,
    device=device,
    save_path=meta_ckpt_path,
    inner_update_batch_size=16,
    validate_every=10,
)
print("✅ Saved best θ★ from Reptile at:", meta_ckpt_path)

# Warm start PPO
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
        train_dataset, actor_critic, optimizer, device=device, update_batch_size_size=32,log_dir=log_dir
    )

    # === Validation: greedy και best-of-K ===
    actor_critic.eval()

    # best-of-1 (stochastic=1 rollout) + greedy
    val_stoch_1, val_greedy = validate_ppo(
        dataloader=dataloaders['val'],
        actor_critic=actor_critic,
        device=device,
        limit_instances=VAL_LIMIT,
        best_of_k=1,
        progress_every=PROGRESS_EVERY,
    )

    # best-of-K 
    if BEST_OF_K > 1:
        val_stoch_K, _ = validate_ppo(
            dataloader=dataloaders['val'],
            actor_critic=actor_critic,
            device=device,
            limit_instances=VAL_LIMIT,
            best_of_k=BEST_OF_K,
            progress_every=PROGRESS_EVERY,
        )
    else:
        val_stoch_K = val_stoch_1

    actor_critic.train()

    # logging 
    log_data.append({
        "epoch": epoch + 1,
        "train_loss": loss,
        "train_best_makespan": train_makespan,
        "val_stoch_best_of_1": val_stoch_1,
        "val_stoch_best_of_k": val_stoch_K,
        "val_greedy": val_greedy,
        "best_of_k": BEST_OF_K,
        "val_limit": VAL_LIMIT,
    })
    save_training_log_csv(log_data, filename=os.path.join(log_dir, "training_log_ppo.csv"))

    # === Επιλογή metric για saving ===
    score_for_saving = val_stoch_K  # μικρότερο = καλύτερο
    if score_for_saving < best_val_makespan:
        best_val_makespan = score_for_saving
        torch.save(actor_critic.state_dict(), os.path.join(model_dir, "best_ppo.pt"))
        print("✅ Saved new best PPO model.")

    

# === Plot Convergence ===
plot_rl_convergence(all_makespans, save_path=os.path.join(plot_dir, "rl_convergence.png"))

# === Gantt Chart for First Instance (greedy rollout) ===
example = instances[0]
actor_critic.eval()
env = JSSPEnvironment(example['times'], example['machines'])  

env.reset()
done = False
while not done:
    available = env.get_available_actions()
    data = prepare_features(env, device)
    logits, _ = actor_critic(data)
    mask = torch.zeros(len(logits), dtype=torch.bool, device=device)
    mask[available] = True
    logits = logits.masked_fill(~mask, -1e10)
    action = torch.argmax(logits).item()
    _, _, done, _ = env.step(action)

plot_gantt_chart(env.extract_job_assignments(), save_path=os.path.join(gantt_dir, "gantt_chart.png"))

# === Run CP on Taillard ===
print("\nRunning CP on Taillard for comparison...")
run_cp_on_taillard()

# === Run RL on Taillard (Hetero) ===
print("\nRunning RL on Taillard benchmark instances (HeteroGIN)...")
subprocess.call([sys.executable, "-m", "eval.main_test_Hetero_ppo"], cwd=base_dir)

# === Merge and Compare CP vs PPO vs Reptile ===
cp_csv = os.path.join(base_dir, "cp", "cp_makespans.csv")
ppo_csv = os.path.join(base_dir, "eval", "taillard_rl_results.csv")
reptile_csv = os.path.join(base_dir, "eval", "taillard_reptile_results.csv")
merged_csv = os.path.join(comparison_dir, "taillard_comparison.csv")
plot_path = os.path.join(plot_dir, "cp_vs_rl_barplot.png")

# Load & de-duplicate on instance_id (keep last)
cp_df = pd.read_csv(cp_csv).drop_duplicates(subset=["instance_id"], keep="last")
ppo_df = pd.read_csv(ppo_csv).drop_duplicates(subset=["instance_id"], keep="last")

# Detect PPO columns (greedy & best-of-K)
greedy_col = "rl_makespan_greedy" if "rl_makespan_greedy" in ppo_df.columns else "rl_makespan"
best_cols = [c for c in ppo_df.columns if c.startswith("rl_makespan_best_of_")]
best_col = best_cols[0] if best_cols else greedy_col

# Keep only the needed columns
ppo_keep = ["instance_id", greedy_col, best_col] if best_col != greedy_col else ["instance_id", greedy_col]

assert cp_df["instance_id"].is_unique, "CP has duplicate instance_id"
assert ppo_df["instance_id"].is_unique, "PPO has duplicate instance_id"

# Merge CP x PPO, assert one-to-one
merged = pd.merge(
    cp_df[["instance_id", "cp_makespan"]],
    ppo_df[ppo_keep],
    on="instance_id",
    how="inner",
    validate="one_to_one",
)

#  merge Reptile results
if os.path.exists(reptile_csv):
    reptile_df = pd.read_csv(reptile_csv).drop_duplicates(subset=["instance_id"], keep="last")

    rep_greedy = "rl_makespan_greedy" if "rl_makespan_greedy" in reptile_df.columns else "rl_makespan"
    rep_best_cols = [c for c in reptile_df.columns if c.startswith("rl_makespan_best_of_")]
    rep_best = rep_best_cols[0] if rep_best_cols else rep_greedy

    # Rename to avoid column-name collisions
    reptile_renamed = reptile_df.rename(columns={
        rep_greedy: "rl_makespan_greedy_reptile",
        rep_best:   "rl_makespan_best_reptile",
    })

    rep_keep = ["instance_id", "rl_makespan_greedy_reptile"]
    if rep_best != rep_greedy:
        rep_keep.append("rl_makespan_best_reptile")

    merged = pd.merge(
        merged,
        reptile_renamed[rep_keep],
        on="instance_id",
        how="left",
        validate="one_to_one",
    )

# Ensure columns unique (defensive)
merged = merged.loc[:, ~merged.columns.duplicated()].copy()

# Compute gaps using numpy (no alignment/reindex)
cp = merged["cp_makespan"].to_numpy()

# PPO gaps
g = merged[greedy_col].to_numpy()
merged["gap_greedy"] = (g - cp) / cp

if best_col != greedy_col:
    b = merged[best_col].to_numpy()
    merged["gap_best"] = (b - cp) / cp
else:
    merged["gap_best"] = merged["gap_greedy"]

# Reptile gaps (if present)
if "rl_makespan_greedy_reptile" in merged.columns:
    gr = merged["rl_makespan_greedy_reptile"].to_numpy()
    merged["gap_reptile_greedy"] = (gr - cp) / cp
if "rl_makespan_best_reptile" in merged.columns:
    br = merged["rl_makespan_best_reptile"].to_numpy()
    merged["gap_reptile_best"] = (br - cp) / cp

merged.to_csv(merged_csv, index=False)
print(f"✅ Saved merged CP vs PPO vs Reptile comparison to {merged_csv}")


# Plot (function auto-detects best-of-K if present)
plot_cp_vs_rl_comparison(merged.rename(columns={greedy_col: "rl_makespan", best_col: best_col}),
                         save_path=plot_path)

print("\n✅ Full pipeline complete: Reptile → PPO → CP vs RL evaluation.")
