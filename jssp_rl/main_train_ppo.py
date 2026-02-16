import os
import sys
import pickle
import subprocess
import pandas as pd
import torch

from data.dataset import JSSPDataset, split_dataset, get_dataloaders
from env.jssp_environment import JSSPEnvironment
from models.Hetero_actor_critic_ppo import ActorCriticPPO
# from models.gnn import GNNWithAttention
from reptile.meta_reptile import reptile_meta_train  

from train.train_ppo import train
from train.validate_ppo import validate_ppo
# from cp.main_cp import run_cp_on_taillard
from utils.features import make_hetero_data
from torch_geometric.data import Data
from utils.logging_utils import (
    plot_rl_convergence,
    plot_gantt_chart,
    save_training_log_csv,
    # plot_cp_vs_rl_comparison
) 
from eval.test_rl_cp_hybrid import run_hybrid_experiment
from utils.paper_results_formatter import generate_paper_table
import time

try:
    from cp.main_cp import run_cp_on_taillard
    CP_AVAILABLE = True
except ImportError:
    print("[Warning] OR-Tools not installed. CP comparison will be skipped. Run 'pip install ortools' to enable.")
    CP_AVAILABLE = False 
from config import (
    lr, num_epochs, batch_size,
    VAL_LIMIT, BEST_OF_K, LOG_BOTH, PROGRESS_EVERY,
    meta_iterations, meta_batch_size, inner_steps, inner_lr, meta_lr,
    inner_update_batch_size, validate_every,
    CURRICULUM_PHASES, ACTIVE_PHASE, RUN_FULL_CURRICULUM
)

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

def run_phase(phase_name):
    info = CURRICULUM_PHASES[phase_name]
    size_N, size_M = info["size"]
    data_file = info["file"]
    
    print(f"\n[Phase Start] >>> STARTING CURRICULUM PHASE: {phase_name} ({size_N}x{size_M}) <<<")
    print(f"[File] Loading data from: {data_file}")

    # === Load Dataset ===
    dataset_path = os.path.join(base_dir, "saved", data_file)
    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset file {dataset_path} not found. Skipping phase.")
        return

    dataset = JSSPDataset(dataset_path)
    # Filter to ensure we only get the intended size (though files are usually size-pure)
    dataset.instances = dataset.filter_by_size((size_N, size_M))
    
    # USER LIMIT: Take exactly 11 instances (10 for training, 1 for validation)
    if len(dataset.instances) > 11:
        dataset.instances = dataset.instances[:11]
    
    print(f"[Data] Instances loaded: {len(dataset)} (10 Train, 1 Val)")

    # 0.95 * 11 = 10.45 -> int(10.45) = 10. This ensures exactly 10 go to train, 1 to val.
    split_dataset(dataset, train_ratio=0.95)
    dataloaders = get_dataloaders(dataset, batch_size=batch_size)

    # === Model Setup ===
    actor_critic = ActorCriticPPO(
        node_input_dim=3,
        gnn_hidden_dim=64,
        gnn_output_dim=32,
        actor_hidden_dim=32,
        critic_hidden_dim=32,
    ).to(device)

    # Load previous best if starting a later phase or continuing a curriculum
    latest_path = os.path.join(model_dir, "best_ppo_latest.pt")
    if os.path.exists(latest_path):
        print(f"[Load] Warm-starting model from: {latest_path}")
        actor_critic.load_state_dict(torch.load(latest_path, map_location=device))


    # === Run Reptile Meta-Learning ===
    meta_ckpt_path = os.path.join(base_dir, "saved", f"meta_best_{phase_name}.pth")
    
    print(f"\n[Step 1] Running Reptile Meta-Training for {phase_name}...")
    actor_critic = reptile_meta_train(
        task_loader=dataloaders['train'],
        actor_critic=actor_critic,
        val_loader=dataloaders['val'],
        meta_iterations=meta_iterations,          
        meta_batch_size=meta_batch_size,           
        inner_steps=inner_steps,               
        inner_lr=inner_lr,
        meta_lr=meta_lr,
        device=device,
        save_path=meta_ckpt_path,
        inner_update_batch_size_size=inner_update_batch_size,
        inner_switch_epoch=1,
        validate_every=1,            
    )

    actor_critic.load_state_dict(torch.load(meta_ckpt_path, map_location=device))
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

    # === PPO Training ===
    print(f"\n[Step 2] Running PPO Training for {phase_name}...")
    best_val_makespan = float("inf")
    all_makespans = []
    log_data = []

    for epoch in range(num_epochs):
        print(f"\n=== {phase_name} | PPO Epoch {epoch+1}/{num_epochs} ===")
        train_dataset = dataset.get_split("train")  
        start_time = time.time()
        train_makespan, loss = train(
            train_dataset, actor_critic, optimizer, device=device, update_batch_size_size=batch_size, log_dir=log_dir
        )
        epoch_time = time.time() - start_time

        actor_critic.eval()
        val_makespan_greedy = validate_ppo(
            dataloader=dataloaders['val'],
            actor_critic=actor_critic,
            device=device,
            limit_instances=VAL_LIMIT,
            best_of_k=1,
            progress_every=PROGRESS_EVERY,
        )

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
            "phase": phase_name,
            "epoch": epoch + 1,
            "train_loss": loss,
            "train_best_makespan": train_makespan,
            "val_makespan_greedy": val_makespan_greedy,
            "val_makespan_best_of_k": val_makespan_best,
            "epoch_time_seconds": epoch_time
        })

        if val_makespan_best < best_val_makespan:
            best_val_makespan = val_makespan_best
            print(f"⭐ New best validation makespan: {best_val_makespan}. Saving checkpoint...")
            torch.save(actor_critic.state_dict(), os.path.join(model_dir, f"best_ppo_{phase_name}.pt"))
            # Save as latest best for curriculum hand-off
            torch.save(actor_critic.state_dict(), os.path.join(model_dir, "best_ppo_latest.pt"))

        # Save latest epoch checkpoint for future use (per user request)
        latest_epoch_path = os.path.join(model_dir, f"latest_ppo_{phase_name}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': actor_critic.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_makespan': val_makespan_best
        }, latest_epoch_path)

        save_training_log_csv(log_data, filename=os.path.join(log_dir, f"training_log_{phase_name}.csv"))

    # === Plots ===
    plot_rl_convergence(all_makespans, save_path=os.path.join(plot_dir, f"rl_convergence_{phase_name}.png"))
    print(f"[Success] Phase {phase_name} Complete.")

if __name__ == "__main__":
    if RUN_FULL_CURRICULUM:
        phases_to_run = ["Phase_1", "Phase_2", "Phase_3"]
    else:
        phases_to_run = [ACTIVE_PHASE]

    for p in phases_to_run:
        run_phase(p)

    # === Final Evaluation (On Taillard) ===
    print("\n[Final Step] Running Benchmarks on Taillard...")
    if CP_AVAILABLE:
        run_cp_on_taillard()
    
    subprocess.call([sys.executable, "-m", "eval.main_test_ppo"], cwd=base_dir)
    
    # === NEW: Hybrid Experiment (RL + CP Warm Start) ===
    print("\n[Step 3] Running Hybrid RL+CP Benchmark...")
    checkpoint = os.path.join(model_dir, "best_ppo_latest.pt")
    taillard_data = os.path.join(base_dir, "saved", "taillard_mixed_instances.pkl")
    
    if os.path.exists(checkpoint) and os.path.exists(taillard_data):
        # We give CP Cold and Hybrid a 20s limit for fair comparison
        # Fix ratio 0.4 handles suggestions for first 40% of ops
        run_hybrid_experiment(checkpoint, taillard_data, time_limit=20, fix_ratio=0.4)
    
    # === NEW: Generate Paper Summary Table ===
    print("\n[Step 4] Generating Paper-Ready Summary Results...")
    generate_paper_table(
        training_logs_dir=log_dir,
        hybrid_results_path=os.path.join(base_dir, "eval", "hybrid_experiment_results.csv"),
        output_path=os.path.join(outputs_dir, "final_paper_results.md")
    )

    print("\n[Finished] Full Curriculum & Research Pipeline Finished.")

