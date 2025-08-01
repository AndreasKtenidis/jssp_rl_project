import torch, pickle, os
from torch.utils.data import DataLoader
from data.dataset import JSSPDataset
from models.actor_critic_ppo import ActorCriticPPO
from meta_reptile_ppo import reptile_meta_train     

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")

# -------- Dataset ----------------------------------------------------------
base = os.path.dirname(__file__)
with open(os.path.join(base, "saved", "synthetic_500_instances.pkl"), "rb") as f:
    instances = pickle.load(f)

dataset = JSSPDataset(instances)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
val_loader   = DataLoader(dataset[:50], batch_size=1)         

# -------- Model ------------------------------------------------------------
actor_critic = ActorCriticPPO(
        node_input_dim=3, gnn_hidden_dim=128, gnn_output_dim=64,
        actor_hidden_dim=64, critic_hidden_dim=64, action_dim=15*15
).to(device)

# -------- Meta-training ----------------------------------------------------
actor_critic = reptile_meta_train(
        train_loader, actor_critic,
        meta_iterations=750,      # outer-loop steps
        meta_batch_size=8,        # tasks per outer update
        inner_steps=5,            # k inner PPO epochs
        inner_lr=3e-4, meta_lr=1e-2,
        device=device, val_loader=val_loader)

# -------- Save Meta-Initialisation -----------------------------------------
os.makedirs(os.path.join(base, "models"), exist_ok=True)
torch.save(actor_critic.state_dict(), os.path.join(base, "models", "meta_reptile_init.pt"))
print("Saved meta-initialisation ✓")
