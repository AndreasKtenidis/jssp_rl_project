import os
import sys
import pickle
import subprocess
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import JSSPDataset
from env.jssp_environment import JSSPEnvironment
from models.Hetero_actor_critic_ppo import ActorCriticPPO
from reptile.meta_reptile_Hetero import reptile_meta_train

from train.train_Hetero_ppo import train
from train.validate_Hetero_ppo import validate_ppo
from cp.main_cp import run_cp_on_taillard
from cp.cp_solver import solve_instance_your_version
from utils.Heterofeatures import prepare_features
from utils.logging_utils import (
    plot_rl_convergence,
    plot_gantt_chart,
    save_training_log_csv,
    plot_cp_vs_rl_comparison,
)

from config import lr, num_epochs, batch_size, VAL_LIMIT, BEST_OF_K, PROGRESS_EVERY

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Deterministic setup ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# === Base Path Setup ===
base_dir = os.path.dirname(__file__)
outputs_dir = os.path.join(base_dir, "outputs")
log_dir = os.path.join(outputs_dir, "logs")
gantt_dir = os.path.join(outputs_dir, "gantt")
plot_dir = os.path.join(outputs_dir, "plots")
model_dir = os.path.join(outputs_dir, "models")
comparison_dir = os.path.join(outputs_dir, "comparisons")
checkpoints_dir = os.path.join(base_dir, "checkpoints")

for d in [log_dir, gantt_dir, plot_dir, model_dir, comparison_dir, checkpoints_dir]:
    os.makedirs(d, exist_ok=True)

# === Load Dataset ===
with open(os.path.join(base_dir, "saved", "Synthetic_instances_15x15_505.pkl"), "rb") as f:
    instances = pickle.load(f)

instances = instances[:120] # Use all 505 instances from the pickle
dataset = JSSPDataset(instances)

taillard_pkl_path = os.path.join(base_dir, "saved", "taillard_instances.pkl")
with open(taillard_pkl_path, "rb") as f:
    taillard_instances = pickle.load(f)[:2]

# === Deterministic train/eval selection ===
rng = np.random.default_rng(SEED)
all_idx = np.arange(len(instances))

# Fixed train subset (100 instances)
train_idx = rng.choice(all_idx, size=100, replace=False)

# Fixed eval subset (3 synthetic instances not in train)
eval_pool = np.setdiff1d(all_idx, train_idx, assume_unique=False)
eval_syn_idx = rng.choice(eval_pool, size=3, replace=False)
assert len(set(train_idx).intersection(set(eval_syn_idx))) == 0, "Eval overlaps train"

# Build train/eval datasets
train_dataset = Subset(dataset, train_idx.tolist())
eval_instances = [instances[i] for i in eval_syn_idx] + taillard_instances
eval_dataset = JSSPDataset(eval_instances)

# Train loader: shuffles order each epoch, but membership is fixed to train_idx
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Eval loader: fixed order and membership for consistent evaluation
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

print(f"train instances: {len(train_dataset)}")
print(f"eval instances:  {len(eval_dataset)}")

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
    task_loader=train_loader,
    actor_critic=actor_critic,
    val_loader=eval_loader,
    meta_iterations=30,
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

#Warm start PPO
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

    # Shuffle training order each epoch (membership fixed to train_idx)
    epoch_perm = rng.permutation(train_idx)
    train_dataset = Subset(dataset, epoch_perm.tolist())

    train_makespan, loss = train(
        train_dataset, actor_critic, optimizer, device=device, update_batch_size_size=32,log_dir=log_dir
    )
    all_makespans.append(train_makespan)

    # === Validation: greedy και best-of-K ===
    actor_critic.eval()

    # best-of-1 (stochastic=1 rollout) + greedy
    val_stoch_1, val_greedy = validate_ppo(
        dataloader=eval_loader,
        actor_critic=actor_critic,
        device=device,
        limit_instances=VAL_LIMIT,
        best_of_k=1,
        progress_every=PROGRESS_EVERY,
    )

    # best-of-K 
    if BEST_OF_K > 1:
        val_stoch_K, _ = validate_ppo(
            dataloader=eval_loader,
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

    # Save checkpoint every epoch (state_dict only)
    torch.save(
        actor_critic.state_dict(),
        os.path.join(checkpoints_dir, f"epoch_{epoch+1}.pt"),
    )

# === CP + RL on the same 5 instances (run once after training) ===
cp_eval_rows = []
for i in eval_syn_idx.tolist():
    inst = instances[i]
    ms, _ = solve_instance_your_version(inst["times"], inst["machines"])
    cp_eval_rows.append({"instance_id": f"synthetic_{i}", "cp_makespan": ms})
for i in range(len(taillard_instances)):
    inst = taillard_instances[i]
    ms, _ = solve_instance_your_version(inst["times"], inst["machines"])
    cp_eval_rows.append({"instance_id": f"taillard_{i}", "cp_makespan": ms})
cp_eval_csv = os.path.join(comparison_dir, "cp_eval_5.csv")
pd.DataFrame(cp_eval_rows).to_csv(cp_eval_csv, index=False)
print(f"✅ Saved 5-instance CP eval to {cp_eval_csv}")

'''
# === Plot Convergence ===
plot_rl_convergence(all_makespans, save_path=os.path.join(plot_dir, "rl_convergence.png"))

# === Run CP on Taillard ===
print("\nRunning CP on Taillard for comparison...")
run_cp_on_taillard()

# === Run RL on Taillard (Hetero) ===
print("\nRunning RL on Taillard benchmark instances (HeteroGIN)...")
subprocess.call([sys.executable, "-m", "eval.main_test_hetero_ppo"], cwd=base_dir)

# === Merge and Compare CP vs PPO vs Reptile ===
cp_csv = os.path.join(base_dir, "cp", "cp_makespans.csv")
ppo_csv = os.path.join(base_dir, "eval", "taillard_rl_results.csv")
reptile_csv = os.path.join(base_dir, "eval", "taillard_reptile_results.csv")
merged_csv = os.path.join(comparison_dir, "taillard_comparison.csv")
plot_path = os.path.join(plot_dir, "cp_vs_rl_barplot.png")

# Load & de-duplicate on instance_id (keep last)
cp_df = pd.read_csv(cp_csv).drop_duplicates(subset=["instance_id"], keep="last")
ppo_df = pd.read_csv(ppo_csv).drop_duplicates(subset=["instance_id"], keep="last")

#sc10 include stochastic best-of-K columns in Taillard comparison
# Detect PPO columns (greedy & stochastic best-of-K)
greedy_col = "rl_makespan_greedy" if "rl_makespan_greedy" in ppo_df.columns else "rl_makespan"
stoch_cols = [
    c for c in ppo_df.columns
    if c.startswith("rl_makespan_best_of_") or c.startswith("rl_makespan_stochastic_best_of_")
]
ppo_keep = ["instance_id", greedy_col] + stoch_cols

# Keep only the needed columns
#ppo_keep = ["instance_id", greedy_col, best_col] if best_col != greedy_col else ["instance_id", greedy_col]

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

for col in stoch_cols:
    gap_name = f"gap_{col.replace('rl_makespan_', '')}"
    merged[gap_name] = (merged[col].to_numpy() - cp) / cp

# Reptile gaps (if present)
if "rl_makespan_greedy_reptile" in merged.columns:
    gr = merged["rl_makespan_greedy_reptile"].to_numpy()
    merged["gap_reptile_greedy"] = (gr - cp) / cp
if "rl_makespan_best_reptile" in merged.columns:
    br = merged["rl_makespan_best_reptile"].to_numpy()
    merged["gap_reptile_best"] = (br - cp) / cp

merged.to_csv(merged_csv, index=False)
print(f"✅ Saved merged CP vs PPO vs Reptile comparison to {merged_csv}")

#sc10 include stochastic best-of-K columns in Taillard comparison
plot_cp_vs_rl_comparison(merged.rename(columns={greedy_col: "rl_makespan"}),
                         save_path=plot_path)

print("\n✅ Full pipeline complete: Reptile → PPO → CP vs RL evaluation.")'''
