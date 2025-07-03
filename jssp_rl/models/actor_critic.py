import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, gnn_output_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(gnn_output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_embeddings, env, mask=None):
        logits = self.net(node_embeddings).squeeze(-1)  # shape: [num_nodes]
        logits = torch.clamp(logits, min=-20, max=20)

        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))

        probs = F.softmax(logits, dim=0)
        return probs, logits
    
    
class Critic(nn.Module):
    def __init__(self, gnn_output_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(gnn_output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_embeddings):
        global_embedding = node_embeddings.mean(dim=0)  # global state
        value = self.net(global_embedding)
        return value.squeeze()
