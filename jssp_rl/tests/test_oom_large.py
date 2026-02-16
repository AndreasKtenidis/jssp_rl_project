import torch
import os
import sys
from models.Hetero_actor_critic_ppo import ActorCriticPPO
from env.jssp_environment import JSSPEnvironment
from utils.features import make_hetero_data

def test_large_instance_memory():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")
    
    # 50 jobs, 20 machines
    num_jobs = 50
    num_machines = 20
    
    times = torch.randint(1, 100, (num_jobs, num_machines)).float()
    machines = torch.stack([torch.randperm(num_machines) for _ in range(num_jobs)])
    
    print(f"Graph Size: {num_jobs * num_machines} nodes")
    
    model = ActorCriticPPO(
        node_input_dim=3,
        gnn_hidden_dim=64,
        gnn_output_dim=32,
        actor_hidden_dim=32,
        critic_hidden_dim=32
    ).to(device)
    
    env = JSSPEnvironment(times, machines, device=device)
    obs = env.reset()
    
    # Run one forward pass
    try:
        data = make_hetero_data(env, device)
        action_probs, value = model(data)
        print("✅ Forward pass successful!")
        print(f"Action Probs Shape: {action_probs.shape}")
        print(f"Value: {value.item()}")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_large_instance_memory()
