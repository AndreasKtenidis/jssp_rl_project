import os, sys, pickle, time
import torch
from torch_geometric.data import HeteroData
from torch.distributions import Categorical

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from models.Hetero_actor_critic_ppo import ActorCriticPPO
from env.jssp_environment import JSSPEnvironment
import eval.reconstruct_critical_eval as rce
rce.DEVICE = torch.device("cpu")
from eval.reconstruct_critical_eval import _build_static_cache, _fast_hetero_data

DEVICE = torch.device("cpu")

@torch.no_grad()
def _profile_rollout(env, model, job_edges, mach_edges, proc_norm):
    env.reset()
    done = False
    step = 0
    t_start = time.time()
    while not done:
        t0 = time.time()
        data = _fast_hetero_data(env, job_edges, mach_edges, proc_norm)
        t_data = time.time() - t0
        
        t0 = time.time()
        logits, _ = model(data)
        t_gnn = time.time() - t0
        
        t0 = time.time()
        avail = env.get_available_actions()
        t_avail = time.time() - t0
        
        if not avail:
            return float("inf")
        mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=DEVICE)
        mask[avail] = True
        logits = logits.masked_fill(~mask, -1e10)
        action = logits.argmax().item()
        
        t0 = time.time()
        _, _, done, _ = env.step(action)
        t_step = time.time() - t0
        
        step += 1
        if step % 1 == 0:
            print(f"Step {step:5d} | data: {t_data:.4f}s | GNN: {t_gnn:.4f}s | Env: {(t_avail+t_step):.4f}s", flush=True)

    return env.get_makespan()

def main():
    print("Loading instance dt_10x100_0...", flush=True)
    with open("saved/DacolTeppan2022_100_100.pkl", "rb") as f:
        instances = pickle.load(f)
    inst = instances[0]

    print("Loading model...", flush=True)
    model = ActorCriticPPO(3, 64, 32, 32, 32).to(DEVICE)
    ckpt = torch.load("outputs/models/best_ppo_latest.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    times_in = inst["times"].clone().detach().to(DEVICE)
    machines_in = inst["machines"].clone().detach().to(DEVICE)
    env = JSSPEnvironment(times_in, machines_in, device=DEVICE, use_shaping_rewards=False)

    print("Building static cache...", flush=True)
    job_edges, mach_edges, proc_norm = _build_static_cache(env)
    
    print("Running 1 greedy rollout...", flush=True)
    _profile_rollout(env, model, job_edges, mach_edges, proc_norm)

if __name__ == "__main__":
    main()
