# utils/train_diagnostics.py
import os
import csv
import torch

def log_train_metrics_to_csv(
    out_dir,
    epoch,
    batch_idx,
    old_log_probs,
    new_log_probs,
    dist_entropy,
    advantages,
    returns,
    values,
    clip_eps=0.2,
    filename="ppo_train_diagnostics.csv"
):
    """Append PPO diagnostics to a CSV file inside out_dir/logs/."""

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, filename)

    # KL divergence
    with torch.no_grad():
        kl = (old_log_probs - new_log_probs).mean().abs().item()
        ratios = torch.exp(new_log_probs - old_log_probs)
        clip_frac = (torch.abs(ratios - 1.0) > clip_eps).float().mean().item()
        entropy = dist_entropy.mean().item()
        adv_std = advantages.std(unbiased=False).item()

        y = returns.detach().view(-1)
        y_hat = values.detach().view(-1)

        # avoid degenerate var for very small batches
        if y.numel() < 2:
            ev = 0.0
        else:
            var_y = torch.var(y, unbiased=False)                 
            if var_y.item() < 1e-8:
                ev = 0.0
            else:
                ev = (1.0 - torch.var(y - y_hat, unbiased=False) / var_y).item()

    # Write header if first time
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "epoch", "batch_idx", "kl", "clip_frac", "entropy", "adv_std", "value_ev"
            ])
        writer.writerow([epoch, batch_idx, kl, clip_frac, entropy, adv_std, ev])

def log_train_hetero_advantages(
    out_dir,
    epoch,
    batch_idx,
    policy_loss,
    value_loss,
    entropy,
    total_loss,
    kl,
    clip_frac,
    ratio_stats,
    advantages,
    returns,
    values,
    rmse_value_raw,
    ent_coef,
    value_coef_eff,
    lr,
    filename="train_hetero_log.csv",
    advantages_filename="train_hetero_advantages.csv",
):
    log_dir = os.path.join(out_dir, "log_train_hetero_advantages")
    os.makedirs(log_dir, exist_ok=True)
    summary_path = os.path.join(log_dir, filename)
    adv_path = os.path.join(log_dir, advantages_filename)

    with torch.no_grad():
        adv = advantages.detach().view(-1)
        ret = returns.detach().view(-1)
        val = values.detach().view(-1)

        def _stats(x):
            if x.numel() == 0:
                return (0.0, 0.0, 0.0, 0.0)
            return (
                x.mean().item(),
                x.std(unbiased=False).item(),
                x.min().item(),
                x.max().item(),
            )

        adv_mean, adv_std, adv_min, adv_max = _stats(adv)
        ret_mean, ret_std, ret_min, ret_max = _stats(ret)
        val_mean, val_std, val_min, val_max = _stats(val)

    r_mean, r_std, r_min, r_max = ratio_stats

    summary_exists = os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not summary_exists:
            writer.writerow([
                "epoch", "batch_idx",
                "policy_loss", "value_loss", "entropy", "total_loss",
                "kl", "clip_frac",
                "adv_mean", "adv_std",
                "ret_mean", "val_mean",
                "rmse_value_raw",
            ])
        writer.writerow([
            epoch, batch_idx,
            float(policy_loss), float(value_loss), float(entropy), float(total_loss),
            float(kl), float(clip_frac),
            adv_mean, adv_std,
            ret_mean, val_mean,
            float(rmse_value_raw),
        ])

    adv_exists = os.path.exists(adv_path)
    with open(adv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not adv_exists:
            writer.writerow(["epoch", "batch_idx", "advantage", "return", "value"])
        for a, r, v in zip(adv.tolist(), ret.tolist(), val.tolist()):
            writer.writerow([epoch, batch_idx, a, r, v])
