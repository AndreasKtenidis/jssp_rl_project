# utils/policy_utils.py
import torch
from torch.distributions import Categorical


def _split_by_batch_1d(x: torch.Tensor, batch_vec: torch.Tensor):
    if x.dim() != 1: x = x.view(-1)
    if batch_vec.dim() != 1: batch_vec = batch_vec.view(-1)
    B = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 0
    counts = torch.bincount(batch_vec, minlength=B)
    assert counts.sum().item() == x.numel(), "batch_vec δεν συμφωνεί με logits μέγεθος."
    return torch.split(x, counts.tolist())

def per_graph_logprob_and_entropy(
    action_logits: torch.Tensor,
    batch_vec: torch.Tensor,
    actions: torch.Tensor,
):
    """
    Returns:
      - logps: Tensor [B]  log-prob του επιλεγμένου action ανά γράφο
      - ents:  Tensor [B]  entropy ανά γράφο
    """
    if action_logits.dim() != 1:
        action_logits = action_logits.view(-1)

    chunks = _split_by_batch_1d(action_logits, batch_vec)  # list length = B
    B = len(chunks)
    assert B == int(actions.shape[0]), f"graphs={B} vs actions={int(actions.shape[0])}"

    logps, ents = [], []
    for i in range(B):
        lg = chunks[i].view(-1)
        if not torch.isfinite(lg).any():
            raise RuntimeError(f"[per_graph] Graph {i}: logits all non-finite. Check masking/indices.")

        # action index ως tensor στον ίδιο device και long dtype
        a = actions[i]
        if a.dtype != torch.long:
            a = a.long()
        if a.device != lg.device:
            a = a.to(lg.device)

        assert 0 <= int(a) < lg.numel(), f"[per_graph] action {int(a)} out of bounds (0..{lg.numel()-1}) for graph {i}"

        dist = Categorical(logits=lg)
        logps.append(dist.log_prob(a))   # 0D tensor
        ents.append(dist.entropy())      # 0D tensor

    return torch.stack(logps), torch.stack(ents)



