# jssp_rl/main_train.py

import pickle
import torch
from data.dataset import JSSPDataset, split_dataset, get_dataloaders
from models.gnn import GNNWithAttention
from models.actor_critic import Actor, Critic
from env.jssp_environment import JSSPEnvironment
from train.train_loop import train_loop
from train.validate import validate
from utils.logging_utils import (
    plot_rl_convergence,
    plot_gantt_chart,
    save_training_log_csv
)

# === Load Data ===
with open("jssp_rl/saved/synthetic_500_instances.pkl", "rb") as f:
    instances = pickle.load(f)

# === Dataset and Dataloader ===
dataset = JSSPDataset(instances)
split_dataset(dataset)
dataloaders = get_dataloaders(dataset, batch_size=16)

# === Create Model ===
gnn = GNNWithAttention(in_channels=3, hidden_dim=128, out_channels=64, num_heads=4)
actor = Actor(gnn_output_dim=64, action_dim=15 * 15)
critic = Critic(gnn_output_dim=64)

# === Dummy edge_index for precedence (job-wise)
num_jobs, num_machines = instances[0]['times'].shape
edge_list = []
for job_id in range(num_jobs):
    for op in range(num_machines - 1):
        u = job_id * num_machines + op
        v = job_id * num_machines + op + 1
        edge_list.append([u, v])
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
edge_weights = torch.ones(edge_index.size(1), dtype=torch.float)

# === Optimizer ===
optimizer = torch.optim.Adam(
    list(gnn.parameters()) + list(actor.parameters()) + list(critic.parameters()),
    lr=1e-3
)

# === Run Training with Validation ===
log_data = []
num_epochs = 50
all_makespans = []
all_losses = []

for epoch in range(num_epochs):
    print(f"\n🔁 Epoch {epoch+1}/{num_epochs}")
    
    episode_makespans, losses = train_loop(
        dataloader=dataloaders['train'],
        gnn=gnn,
        actor=actor,
        critic=critic,
        edge_index=edge_index,
        edge_weights=edge_weights,
        optimizer=optimizer,
        epochs=1  # Train 1 epoch at a time
    )

    val_makespan = validate(
        dataloaders['val'],
        gnn, actor, critic,
        edge_index, edge_weights
    )

    log_data.append({
        "epoch": epoch + 1,
        "train_loss": losses[-1],
        "train_best_makespan": episode_makespans[-1],
        "val_makespan": val_makespan
    })

    all_makespans.append(episode_makespans[-1])
    all_losses.append(losses[-1])

    save_training_log_csv(log_data, filename="training_log.csv")

# === Plot Convergence ===
plot_rl_convergence(all_makespans, save_path="rl_convergence.png")

# === Plot Gantt Chart (replay one instance for visualization)
env = JSSPEnvironment(instances[0]['times'], instances[0]['machines'])
env.reset()

done = False
while not done:
    x = env.extract_job_assignments()  # You could simulate policy here if needed
    # Just schedule the first operation arbitrarily for demo
    for job in range(env.num_jobs):
        for op in range(env.num_machines):
            if env.state[job, op] == 0 and (op == 0 or env.state[job, op - 1] == 1):
                idx = job * env.num_machines + op
                _, _, done, _ = env.step(idx)
                break
        if done:
            break

plot_gantt_chart(env.extract_job_assignments(), save_path="gantt_chart.html")
