import torch
import torch.nn as nn
from models.gnn import GNNWithAttention  
from torch_geometric.nn import global_mean_pool

class ActorCriticPPO(nn.Module):
    def __init__(self, node_input_dim, gnn_hidden_dim, gnn_output_dim,
                 actor_hidden_dim, critic_hidden_dim, action_dim):
        super(ActorCriticPPO, self).__init__()

        # Shared GNN backbone
        self.gnn = GNNWithAttention(
            in_channels=node_input_dim,
            hidden_dim=gnn_hidden_dim,
            out_channels=gnn_output_dim
        )

        # Actor head
        self.actor_mlp = nn.Sequential(
            nn.Linear(gnn_output_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, 1)  
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
            data: torch_geometric.data.Data or torch_geometric.data.Batch
        Returns:
            action_logits: [num_nodes]
            state_values: [num_graphs, 1]
        """
        node_embeddings = self.gnn(data.x, data.edge_index)

        # Actor: output one logit per node
        action_logits = self.actor_mlp(node_embeddings).squeeze(-1)

        # Critic: pool node embeddings to graph embeddings
        if hasattr(data, 'batch'):
            graph_embeddings = global_mean_pool(node_embeddings, data.batch)
        else:
            # In single-graph case (rollout), treat all nodes as one graph
            batch = torch.zeros(node_embeddings.size(0), dtype=torch.long, device=node_embeddings.device)
            graph_embeddings = global_mean_pool(node_embeddings, batch)

        state_values = self.critic_mlp(graph_embeddings)  # [num_graphs, 1]
        return action_logits, state_values

    def act(self, data, mask=None):
        """
        Selects an action from the policy πθ.
        Uses stochastic sampling during training and greedy (argmax) during evaluation.
        Returns:
            action: int
            log_prob: float tensor
            value: float tensor
        """
        action_logits, state_values = self.forward(data)

        if mask is not None:
            action_logits[~mask] = -1e10

        action_dist = torch.distributions.Categorical(logits=action_logits)

        if self.training:
            # Stochastic sample for exploration
            action = action_dist.sample()
        else:
            # Greedy selection during evaluation
            action = torch.argmax(action_logits)

        log_prob = action_dist.log_prob(action)
        value = state_values.squeeze(0)

        return action.item(), log_prob, value


    def evaluate_actions(self, data, actions, mask=None):
        """
        Used during PPO update. Evaluates log_probs, entropy, and values.
        """
        action_logits, state_values = self.forward(data)

        if mask is not None:
            action_logits[~mask] = -1e10

        
        action_logits = action_logits.squeeze()
        state_values = state_values.squeeze(-1)

        action_dist = torch.distributions.Categorical(logits=action_logits)

        
        if isinstance(actions, list):
            actions = torch.tensor(actions, dtype=torch.long, device=action_logits.device)
        else:
            actions = actions.to(dtype=torch.long)

        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()

        return log_probs, entropy, state_values

