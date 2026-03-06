import os
import pickle
import torch
import pandas as pd
import time
from models.Hetero_actor_critic_ppo import ActorCriticPPO
from env.jssp_environment import JSSPEnvironment
from cp.cp_solver import solve_with_rl_warmstart, solve_instance_your_version
from utils.features import make_hetero_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def get_rl_schedule(instance, model):
    times = instance['times'] if isinstance(instance['times'], torch.Tensor) else torch.tensor(instance['times'])
    machines = instance['machines'] if isinstance(instance['machines'], torch.Tensor) else torch.tensor(instance['machines'])
    times, machines = times.to(device), machines.to(device)
    env = JSSPEnvironment(times, machines, device=device, use_shaping_rewards=False)
    env.reset()
    
    # Pre-calculate static graph topology once per instance
    from models.gin import HeteroGATv2
    static_edge_index = HeteroGATv2.build_edge_index_dict(env.machines)

    done = False
    ops = 0
    total_ops = env.num_jobs * env.num_machines
    while not done:
        data = make_hetero_data(env, device, precomputed_edge_index=static_edge_index)

        logits, _ = model(data)
        avail = env.get_available_actions()
        mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=device)
        mask[avail] = True
        logits = logits.masked_fill(~mask, -1e10)
        action = torch.argmax(logits).item()
        
        _, _, done, _ = env.step(action)
        ops += 1
        if ops % 100 == 0:
            print(".", end="", flush=True)

    print(f" (RL Done)", flush=True)
    return env.get_makespan(), env.extract_job_assignments()

def run_scaling_test():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    ckpt = os.path.join(base_dir, "outputs", "models", "best_ppo_Phase_5.pt")
    if not os.path.exists(ckpt):
        print("Checkpoint not found.")
        return

    model = ActorCriticPPO(3, 64, 32, 32, 32).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    model.eval()

    with open(os.path.join(base_dir, "saved", "benchmark_taillard.pkl"), "rb") as f:
        instances = pickle.load(f)

    # Test indices for 20x20 (ta21 to ta23)
    test_indices = [20, 21, 22] 
    results = []

    for idx in test_indices:
        inst = instances[idx]
        name = inst['name']
        size = inst['times'].shape
        tl = 30.0 # Time limit for 20x20 as per our adaptive strategy
        
        print(f"\nEvaluating {name} ({size[0]}x{size[1]}) | TL={tl}s")
        
        # RL
        rl_ms, rl_sched = get_rl_schedule(inst, model)
        print(f"  RL Makespan: {rl_ms}")

        # Pure CP
        print(f"  Pure CP solving...")
        start_cp = time.time()
        cp_ms, _ = solve_instance_your_version(inst['times'], inst['machines'], time_limit_s=tl)
        cp_time = time.time() - start_cp
        print(f"  Pure CP MS: {cp_ms} in {cp_time:.2f}s")

        # Hybrid (20%)
        print(f"  Hybrid (20%) solving...")
        start_h = time.time()
        h_ms, _ = solve_with_rl_warmstart(inst['times'], inst['machines'], rl_sched, time_limit_s=tl, fix_ratio=0.2)
        h_time = time.time() - start_h
        print(f"  Hybrid MS: {h_ms} in {h_time:.2f}s")

        results.append({
            "name": name,
            "size": f"{size[0]}x{size[1]}",
            "rl_ms": rl_ms,
            "cp_ms": cp_ms if cp_ms else "N/A",
            "cp_time": round(cp_time, 2),
            "hybrid_ms": h_ms,
            "hybrid_time": round(h_time, 2)
        })

    df = pd.DataFrame(results)
    print("\nScaling Test Results:")
    print(df.to_string())
    df.to_csv(os.path.join(base_dir, "eval", "scaling_effect_results.csv"), index=False)

if __name__ == "__main__":
    run_scaling_test()
