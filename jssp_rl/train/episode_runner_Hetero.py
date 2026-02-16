# jssp_rl/train/episode_runner.py

import torch
from utils.Heterofeatures import prepare_features
from utils.action_masking import select_top_k_actions

def run_episode(env, actor_critic, epsilon=0.1, device=None):
    log_probs, values, rewards = [], [], []
    env.reset()
    done = False

    while not done:
        # Build HeteroData input
        data = prepare_features(env, device)

        # Mask valid operations only
        valid_actions, top_k_mask = select_top_k_actions(env)

        # Get action, log_prob, value
        with torch.no_grad():
            action, log_prob, value = actor_critic.act(data, mask=top_k_mask.to(device))

        # Environment step
        _, reward, done, _ = env.step(action)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor(reward, dtype=torch.float32).to(device))

    return log_probs, values, rewards
