# jssp_rl/train/episode_runner.py

import torch
from utils.features import prepare_features
from utils.action_masking import select_top_k_actions

def run_episode(env, gnn, actor, critic, edge_index, edge_weights, epsilon=0.1, device=None, current_epoch=0, switch_epoch=10):
    

    log_probs, values, rewards = [], [], []
    env.reset()
    done = False

    while not done:
        x = prepare_features(env, edge_index,device=device)
        node_embeddings = gnn(x, edge_index.to(device), edge_weights.to(device))
        valid_actions, top_k_mask = select_top_k_actions(env)

        probs, _ = actor(node_embeddings, env, mask=top_k_mask)  
        value = critic(node_embeddings, env)

        if torch.rand(1).item() < epsilon:
            action = valid_actions[torch.randint(len(valid_actions), (1,))].item()
        else:
            action = torch.multinomial(probs, 1).item()

        log_prob = torch.log(probs[action] + 1e-6)

        _, reward, done, _ = env.step(action)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor(reward, dtype=torch.float32).to(device))

    return log_probs, values, rewards

        


