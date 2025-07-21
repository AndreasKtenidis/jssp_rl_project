import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


from utils.buffer import RolloutBuffer
from config import clip_epsilon, gamma, gae_lambda, lr, ppo_epochs, batch_size ,value_coef,entropy_coef


def train(env, actor_critic, device):
    buffer = RolloutBuffer()
    actor_critic.to(device)
    optimizer = optim.Adam(actor_critic.parameters(), lr=lr)

    # ----- PHASE 1: COLLECT TRAJECTORIES -----
    state = env.reset()
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).flatten().unsqueeze(0)
        available_actions = env.get_available_actions()

        with torch.no_grad():
            action_logits, value = actor_critic(state_tensor)
            mask = torch.zeros_like(action_logits)
            mask[0, available_actions] = 1  # Mask only valid actions
            masked_logits = action_logits.masked_fill(mask == 0, float('-inf'))
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        next_state, reward, done, _ = env.step(action.item())

        buffer.add(state_tensor, action, reward, log_prob, value, done)
        state = next_state

    # ----- PHASE 2: COMPUTE ADVANTAGES -----
    with torch.no_grad():
        _, last_value = actor_critic(torch.tensor(state, dtype=torch.float32, device=device).flatten().unsqueeze(0))
    buffer.compute_returns_and_advantages(last_value.squeeze(), gamma, gae_lambda)

    # ----- PHASE 3: POLICY & VALUE UPDATE -----
    for _ in range(ppo_epochs):
        for batch in buffer.get_batches(batch_size):
            states, actions, old_log_probs, returns, advantages = batch

            action_logits, values = actor_critic(states)
            dist = Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = (new_log_probs - old_log_probs).exp()
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = nn.MSELoss()(values.squeeze(), returns)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    buffer.clear()
