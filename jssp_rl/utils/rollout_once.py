# utils/rollout_once.py
import torch
from utils.Heterofeatures import prepare_features
from utils.action_masking import select_top_k_actions
from utils.HeteroBuffer import RolloutBuffer
from config import gamma, gae_lambda


@torch.no_grad()
def rollout_once(env, model, device, use_est_boost, est_beta, mode="stochastic",
                  max_steps=None):
    """
    Roll out one episode.

    mode:
      - "stochastic": collect transitions into RolloutBuffer for PPO
      - "greedy":  run episode, no buffer, just stats

    Returns:
      if mode="stochastic":
        (RolloutBuffer, stats)
      if mode="greedy":
        stats
    """
    if mode not in ["stochastic", "greedy"]:
        raise ValueError("mode must be 'stochastic' or 'greedy'")

    # για eval κρατάμε dropout off (eval mode), για train sample on
    if mode == "stochastic":
        model.train()
    elif mode == "greedy" :
        model.eval()
    else:
        raise ValueError("mode must be 'stochastic' or 'greedy'")
    
    env.reset()
    done = False
    steps = 0
    total_reward = 0.0
    one_legal_counter = 0

    ep_buf = RolloutBuffer() if mode == "stochastic" else None

    while not done and (max_steps is None or steps < max_steps):
        data = prepare_features(env, device)

        # unified legal mask
        _, valid_mask, _, _ = select_top_k_actions(env, top_k=None, device=device)
        assert valid_mask.numel() == data['op'].x.size(0), \
            f"[MaskLen] mask={valid_mask.numel()} vs nodes={data['op'].x.size(0)}"

        if valid_mask.sum().item() == 0:
            print(f"[ROLL] No legal actions at t={steps}, aborting episode")
            break
        if valid_mask.sum().item() == 1:
            one_legal_counter += 1

        # act
        action, log_prob, value = model.act(
            data,
            use_est_boost=use_est_boost,
            est_beta=est_beta,
            mask=valid_mask,
            env=env if use_est_boost else None
        )

        # step
        _, reward, done, _ = env.step(action)
        total_reward = reward

        if mode == "stochastic":
            # store transition
            ep_buf.add(
                state=data,
                action=action,
                reward=reward,
                log_prob=log_prob,
                value=value.view(-1),
                done=done,
                mask=valid_mask,
                topk_mask=torch.zeros_like(valid_mask, dtype=torch.bool, device=device)
            )

        steps += 1

    # final bootstrapping μόνο για training
    if mode == "stochastic":
        final_data = prepare_features(env, device)
        _, final_value = model(final_data)
        ep_buf.compute_returns_and_advantages(final_value.squeeze(),
                                              gamma=gamma, gae_lambda=gae_lambda)

    stats = dict(
        makespan=env.get_makespan(),
        total_reward=total_reward,
        steps=steps,
        one_legal_counter=one_legal_counter,
        finished=done,
        env=env
    )

    if mode == "stochastic":
        return ep_buf, stats
    else:
        return stats
