"""main_test_ppo.py - Evaluate the best PPO model on Taillard benchmark
======================================================================
* Loads the checkpoint saved as best_ppo.pt (from main_ppo.py)
* Runs greedy policy (argmax over masked logits) on every Taillard instance
* Saves makespans + gantt charts + CSV.
"""

import os
import pickle
import torch
import pandas as pd
from models.gnn import GNNWithAttention
from models.actor_critic_ppo import ActorCriticPPO
from env.jssp_environment import JSSPEnvironment
from utils.logging_utils import plot_gantt_chart

# ── Device ──────────────────────────────────────────────────────────
device = torch.device("cpu" ) #if torch.cuda.is_available() else "cpu")


# ── Helper --------------------------------------------------------------------

def load_checkpoint(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No PPO checkpoint found at {path}. Train the model first.")
    state = torch.load(path, map_location=device)
    return state

@torch.no_grad()
def run_rl_on_instance(env, model):
    """Greedy rollout using PPO actor head with masking."""
    state = env.reset()
    done = False

    while not done:
        # flatten state into 1‑D tensor (same as training)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).flatten().unsqueeze(0)
        logits, _ = model(state_tensor)

        # mask illegal operations
        avail = env.get_available_actions()
        mask = torch.zeros_like(logits)
        mask[0, avail] = 1
        logits = logits.masked_fill(mask == 0, -1e9)

        action = torch.argmax(logits, dim=-1).item()
        state, _, done, _ = env.step(action)

    return env.get_makespan(), env.extract_job_assignments()

# ── Main ----------------------------------------------------------------------

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__)) 
    taillard_pkl = os.path.join(base_dir, "saved", "taillard_instances.pkl")
    checkpoint   = os.path.join(base_dir, "best_ppo.pt")
    results_csv  = os.path.join(base_dir, "eval", "taillard_rl_results.csv")
    gantt_dir    = os.path.join(base_dir, "eval", "gantt_rl")
    os.makedirs(gantt_dir, exist_ok=True)

    # ── Load benchmark instances ──────────────────────────────────
    with open(taillard_pkl, "rb") as f:
        instances = pickle.load(f)

    # ── Build model & load checkpoint ─────────────────────────────
    gnn = GNNWithAttention(in_channels=3, hidden_dim=128, out_channels=64, num_heads=4).to(device)
    model = ActorCriticPPO(gnn=gnn, action_dim=15*15).to(device)

    state_dict = load_checkpoint(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # ── Evaluate each Taillard instance ───────────────────────────
    results = []
    for idx, inst in enumerate(instances):
        print(f"🧪 PPO testing on Taillard instance {idx+1}/{len(instances)}")
        times = torch.tensor(inst["times"], dtype=torch.float32)
        machines = torch.tensor(inst["machines"], dtype=torch.long)
        env = JSSPEnvironment(times, machines, device=device)

        makespan, schedule = run_rl_on_instance(env, model)
        results.append({"instance_id": idx, "rl_makespan": makespan})

        gantt_path = os.path.join(gantt_dir, f"gantt_rl_taillard_{idx:02d}.png")
        plot_gantt_chart(schedule, save_path=gantt_path, show_op_index=True)

    pd.DataFrame(results).to_csv(results_csv, index=False)
    print(f"✅ PPO test results saved to {results_csv}")

if __name__ == "__main__":
    main()
