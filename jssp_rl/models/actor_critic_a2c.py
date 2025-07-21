import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, gnn_output_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(gnn_output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_embeddings, mask=None):
        logits = self.net(node_embeddings).squeeze(-1)  
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

    def forward(self, node_embeddings, env):
        
        scheduled_mask = (env.state.flatten() == 1)

        if scheduled_mask.sum() == 0:
            global_embedding = node_embeddings.mean(dim=0)
        else:
            global_embedding = node_embeddings[scheduled_mask].mean(dim=0)

        value = self.net(global_embedding)
        return value.squeeze()

