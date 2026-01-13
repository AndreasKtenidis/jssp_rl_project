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
