# action_masking.py
import torch

def select_top_k_actions(env, top_k: int = None, device=None):
    """
    Returns:
      - valid_indices: LongTensor [Nv] with indices of all legal actions
      - valid_mask:    BoolTensor [num_ops] (True = legal)
      - topk_indices:  LongTensor [K'] with the selected top-k (subset of legal)
      - top_k_mask:    BoolTensor [num_ops] (True = in the top-k set)

    Notes:
      * valid_mask is always all legal ops (hard mask).
      * top_k_mask is optional "soft" guidance (bias).
      * If top_k is None or <= 0, we disable top-k selection and return empty topk_indices/mask.
    """
    num_ops = env.num_jobs * env.num_machines
    valid_mask = torch.zeros(num_ops, dtype=torch.bool, device=device)
    priorities = []  # (est, -proc_time, flat_index)

    for job_id in range(env.num_jobs):
        for op_idx in range(env.num_machines):
            idx = job_id * env.num_machines + op_idx

            if env.state[job_id, op_idx] == 0 and (op_idx == 0 or env.state[job_id, op_idx - 1] == 1):
                est = env.job_completion_times[job_id, op_idx - 1] if op_idx > 0 else 0
                pt  = env.times[job_id, op_idx]
                valid_mask[idx] = True
                pt_val = pt.item() if torch.is_tensor(pt) else pt
                priorities.append((est, -pt_val, idx))

            if env.state[job_id, op_idx] == 0:
                break

    if len(priorities) == 0:
        empty = torch.tensor([], dtype=torch.long, device=device)
        top_k_mask = torch.zeros(num_ops, dtype=torch.bool, device=device)
        return empty, valid_mask, empty, top_k_mask

    # --- None top-k guidance ---
    if top_k is None or top_k <= 0:
        empty = torch.tensor([], dtype=torch.long, device=device)
        top_k_mask = torch.zeros(num_ops, dtype=torch.bool, device=device)
        valid_indices = valid_mask.nonzero(as_tuple=False).view(-1)
        return valid_indices, valid_mask, empty, top_k_mask

    # --- top-k choice ---
    priorities.sort(key=lambda x: (x[0], x[1]))
    take = min(top_k, len(priorities))
    sel = [idx for *_, idx in priorities[:take]]
    topk_indices = torch.tensor(sel, dtype=torch.long, device=device)

    top_k_mask = torch.zeros_like(valid_mask)
    top_k_mask[topk_indices] = True

    valid_indices = valid_mask.nonzero(as_tuple=False).view(-1)
    return valid_indices, valid_mask, topk_indices, top_k_mask
