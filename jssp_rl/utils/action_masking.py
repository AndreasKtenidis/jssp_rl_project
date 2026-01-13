# action_masking.py
import torch

def select_top_k_actions(env, top_k: int = 10, device=None):
    """
    Returns:
      - valid_indices: LongTensor [Nv] with indices of all legal actions
      - valid_mask:    BoolTensor [num_ops] (True = legal)
      - topk_indices:  LongTensor [K'] with the selected top-k (subset of legal)
      - top_k_mask:    BoolTensor [num_ops] (True = in the top-k set)

    Notes:
      * We ONLY use valid_mask for hard masking the policy logits.
      * top_k_mask is optional "soft" guidance (bias), not a hard mask.
      * We break at the first unprocessed op per job (matches env.get_available_actions()).
    """
    num_ops = env.num_jobs * env.num_machines
    valid_mask = torch.zeros(num_ops, dtype=torch.bool, device=device)
    priorities = []  # list of tuples (earliest_start_time, -proc_time, flat_index)

    for job_id in range(env.num_jobs):
        for op_idx in range(env.num_machines):
            idx = job_id * env.num_machines + op_idx

            # legality: operation not done, and predecessor done (or first op)
            if env.state[job_id, op_idx] == 0 and (op_idx == 0 or env.state[job_id, op_idx - 1] == 1):
                est = env.job_completion_times[job_id, op_idx - 1] if op_idx > 0 else 0
                pt  = env.times[job_id, op_idx]
                valid_mask[idx] = True

                # priority: earlier est first; tie-break by longer processing time (-pt)
                pt_val = pt.item() if torch.is_tensor(pt) else pt
                priorities.append((est, -pt_val, idx))

            # IMPORTANT: stop at the first not-yet-done op per job
            # (this mirrors env.get_available_actions() logic)
            if env.state[job_id, op_idx] == 0:
                break

    if len(priorities) == 0:
        # no legal actions
        empty = torch.tensor([], dtype=torch.long, device=device)
        top_k_mask = torch.zeros(num_ops, dtype=torch.bool, device=device)
        return empty, valid_mask, empty, top_k_mask

    # pick top-k by (est asc, proc_time desc)
    priorities.sort(key=lambda x: (x[0], x[1]))
    take = min(top_k, len(priorities))
    sel = [idx for *_, idx in priorities[:take]]
    topk_indices = torch.tensor(sel, dtype=torch.long, device=device)

    # build the boolean mask for the top-k set
    top_k_mask = torch.zeros_like(valid_mask)
    top_k_mask[topk_indices] = True

    # list of all legal action indices
    valid_indices = valid_mask.nonzero(as_tuple=False).view(-1)
    return valid_indices, valid_mask, topk_indices, top_k_mask
