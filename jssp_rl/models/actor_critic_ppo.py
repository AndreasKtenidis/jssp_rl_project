import torch
import torch.nn as nn
from gnn import GNNWithAttention  

class ActorCriticPPO(nn.Module):
    def __init__(self, node_input_dim, gnn_hidden_dim, gnn_output_dim,
                 actor_hidden_dim, critic_hidden_dim, action_dim):
        super(ActorCriticPPO, self).__init__()

        # Shared GNN backbone
        self.gnn = GNNWithAttention(
            node_input_dim=node_input_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim
        )

        # Actor head
        self.actor_mlp = nn.Sequential(
            nn.Linear(gnn_output_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, action_dim)
        )

        # Critic head
        self.critic_mlp = nn.Sequential(
            nn.Linear(gnn_output_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, 1)
        )

    def forward(self, data):
        """
        Args:
            data: torch_geometric.data.Data
        Returns:
            action_logits: [num_nodes, action_dim]
            state_values: [num_nodes, 1]
        """
        node_embeddings = self.gnn(data)  # [num_nodes, gnn_output_dim]
        action_logits = self.actor_mlp(node_embeddings)
        state_values = self.critic_mlp(node_embeddings)
        return action_logits, state_values

    def act(self, data, mask=None):
        """
        Samples an action from the policy πθ.
        Returns action, log_prob, value
        """
        action_logits, state_values = self.forward(data)
        if mask is not None:
            action_logits[~mask] = -1e10

        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob, state_values.squeeze(-1)

    def evaluate_actions(self, data, actions, mask=None):
        """
        Used during PPO update. Evaluates log_probs, entropy, and values.
        """
        action_logits, state_values = self.forward(data)
        if mask is not None:
            action_logits[~mask] = -1e10

        action_dist = torch.distributions.Categorical(logits=action_logits)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()

        return log_probs, entropy, state_values.squeeze(-1)
