# S# jssp_rl/train/episode_runner.py

import torch
import torch.nn.functional as F
from utils.features import prepare_features
from utils.action_masking import select_top_k_actions


def run_episode(env, gnn, actor, critic, edge_index, edge_weights, epsilon=0.1):
    log_probs, values, rewards = [], [], []

    env.reset()
    done = False

    while not done:
        x = prepare_features(env, edge_index)
        node_embeddings = gnn(x, edge_index, edge_weights)
        probs, _ = actor(node_embeddings, env)
        value = critic(node_embeddings)

        valid_actions, top_k_mask = select_top_k_actions(env)

        # Epsilon-greedy
        if torch.rand(1).item() < epsilon:
            action = valid_actions[torch.randint(len(valid_actions), (1,))].item()
        else:
            masked_probs = probs * top_k_mask.float()
            masked_probs = masked_probs / (masked_probs.sum() + 1e-8)
            action = torch.multinomial(masked_probs, 1).item()

        log_prob = torch.log(probs[action] + 1e-6)

        _, reward, done, _ = env.step(action)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor(reward, dtype=torch.float32))

    return log_probs, values, rewards

