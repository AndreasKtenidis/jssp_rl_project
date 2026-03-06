import torch
from torch_geometric.data import HeteroData, Batch


class RolloutBuffer:
    """
    Memory-efficient RolloutBuffer.
    
    Instead of storing full HeteroData objects (which include static edges
    and waste GBs of GPU/CPU RAM for large instances like 50x20 or 100x20),
    we store ONLY the lightweight node feature tensors (shape [N, 3]).
    
    The static edge_index_dict is stored once per episode and reused
    when reconstructing HeteroData during the PPO update step.
    """

    def __init__(self):
        self.node_features = []    # list[Tensor shape (N, 3)]  <- only features, no edges
        self.edge_index_dicts = [] # list[dict]  <- one per transition (shared reference, cheap)
        self.actions = []          # list[int]
        self.rewards = []          # list[float]
        self.log_probs = []        # list[tensor scalar]
        self.values = []           # list[tensor scalar]
        self.dones = []            # list[bool]
        self.masks = []            # list[BoolTensor shape [N]]

        self.returns = []
        self.advantages = []

    def clear(self):
        self.__init__()

    def add(self, node_x: torch.Tensor, edge_index_dict: dict,
            action: int, reward: float, log_prob: torch.Tensor,
            value: torch.Tensor, done: bool, mask: torch.Tensor):
        """
        Add a single transition.
        
        Args:
            node_x: Node feature tensor [N, F] (CPU is fine, we move to device at update)
            edge_index_dict: Static edge dict for this instance (shared reference)
        """
        self.node_features.append(node_x.detach().cpu())     # store on CPU to save VRAM
        self.edge_index_dicts.append(edge_index_dict)          # just a reference, near-zero cost
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.log_probs.append(log_prob.detach().cpu())
        self.values.append(value.detach().cpu())
        self.dones.append(bool(done))
        self.masks.append(mask.detach().cpu().to(dtype=torch.bool))

    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0
        last_val = last_value.item() if hasattr(last_value, 'item') else float(last_value)
        values_ext = [v.item() for v in self.values] + [last_val]

        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values_ext[step + 1] * (1 - self.dones[step]) - values_ext[step]
            gae = delta + gamma * lam * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)

        self.advantages = [torch.tensor(a, dtype=torch.float32) for a in advantages]
        self.returns = [adv + val for adv, val in zip(self.advantages, self.values)]

    def _rebuild_hetero_data(self, idx: int, device) -> HeteroData:
        """Reconstruct a HeteroData object from stored features + edge_index_dict."""
        data = HeteroData()
        data['op'].x = self.node_features[idx].to(device)
        eid = self.edge_index_dicts[idx]
        data['op', 'job', 'op'].edge_index = eid[('op', 'job', 'op')].to(device)
        data['op', 'machine', 'op'].edge_index = eid[('op', 'machine', 'op')].to(device)
        return data

    def get_batches(self, batch_size: int, device=None):
        """
        Yield mini-batches for PPO updates.
        
        Reconstructs HeteroData on-the-fly from stored features + edges.
        """
        total_size = len(self.node_features)
        if total_size == 0:
            return
        if device is None:
            device = self.node_features[0].device if self.node_features else torch.device('cpu')

        indices = torch.randperm(total_size)

        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]

            # Reconstruct HeteroData objects and batch them
            data_list = [self._rebuild_hetero_data(i.item(), device) for i in batch_indices]
            batch_data: Batch = Batch.from_data_list(data_list)

            # Flat scalar tensors
            actions_batch       = torch.tensor([self.actions[i]    for i in batch_indices], dtype=torch.long, device=device)
            old_log_probs_batch = torch.stack([self.log_probs[i]   for i in batch_indices]).view(-1).to(device)
            returns_batch       = torch.stack([self.returns[i]     for i in batch_indices]).view(-1).to(device)
            advantages_batch    = torch.stack([self.advantages[i]  for i in batch_indices]).view(-1).to(device)
            old_values_batch    = torch.stack([self.values[i]      for i in batch_indices]).view(-1).to(device)

            # Concatenate masks
            masks_batch = torch.cat([self.masks[i] for i in batch_indices], dim=0).to(device)

            # Sanity check
            num_nodes_total = int(batch_data['op'].x.size(0))
            if masks_batch.numel() != num_nodes_total:
                raise RuntimeError(
                    f"[Buffer] masks_batch length ({masks_batch.numel()}) "
                    f"!= num_nodes_in_batch ({num_nodes_total})."
                )

            yield (
                batch_data,
                actions_batch,
                old_log_probs_batch,
                returns_batch,
                advantages_batch,
                old_values_batch,
                masks_batch
            )

    def merge(self, other: "RolloutBuffer"):
        self.node_features     += other.node_features
        self.edge_index_dicts  += other.edge_index_dicts
        self.actions           += other.actions
        self.rewards           += other.rewards
        self.log_probs         += other.log_probs
        self.values            += other.values
        self.dones             += other.dones
        self.masks             += other.masks

        if getattr(other, "returns", None):
            self.returns    += other.returns
        if getattr(other, "advantages", None):
            self.advantages += other.advantages
