import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from models.gin import HeteroGATv2, HeteroGAT, HeteroGIN

class ActorCriticPPO(nn.Module):
    def __init__(self, node_input_dim, gnn_hidden_dim, gnn_output_dim,
                 actor_hidden_dim, critic_hidden_dim):
        super(ActorCriticPPO, self).__init__()

        self.gnn = HeteroGATv2(
        #sc4 self.gnn = HeteroGAT(
        #sc4 self.gnn = HeteroGIN(
            in_channels=node_input_dim,
            hidden_dim=gnn_hidden_dim,
            out_channels=gnn_output_dim
        )

        self.actor_mlp = nn.Sequential(
            nn.Linear(gnn_output_dim, actor_hidden_dim),
            nn.GELU(),
            #sc1 nn.ReLU(),
            nn.Linear(actor_hidden_dim, 1)
        )

        self.critic_mlp = nn.Sequential(
            nn.Linear(gnn_output_dim, critic_hidden_dim),
            nn.GELU(),
            #sc1 nn.ReLU(),
            nn.Linear(critic_hidden_dim, 1)
        )

    def forward(self, data):
        node_embeddings = self.gnn(data.x_dict, data.edge_index_dict)

        action_logits = self.actor_mlp(node_embeddings).squeeze(-1)

        batch = getattr(data['op'], 'batch', None)
        if batch is None:
            batch = torch.zeros(node_embeddings.size(0), dtype=torch.long, device=node_embeddings.device)

        graph_embeddings = global_mean_pool(node_embeddings, batch)
        state_values = self.critic_mlp(graph_embeddings)
        return action_logits, state_values

    def act(self, data, mask=None):
        action_logits, state_values = self.forward(data)

        if mask is not None:
            action_logits[~mask] = -1e10

        action_dist = torch.distributions.Categorical(logits=action_logits)

        if self.training:
            action = action_dist.sample()
        else:
            action = torch.argmax(action_logits)

        log_prob = action_dist.log_prob(action).detach()
        value = state_values.squeeze(0).detach()
        return action.item(), log_prob, value

    def evaluate_actions(self, data, actions, mask=None):
        action_logits, state_values = self.forward(data)

        if mask is not None:
            action_logits[~mask] = -1e10

        state_values = state_values.squeeze(-1)

        action_dist = torch.distributions.Categorical(logits=action_logits)

        if isinstance(actions, list):
            actions = torch.tensor(actions, dtype=torch.long, device=action_logits.device)
        else:
            actions = actions.to(dtype=torch.long)

        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()

        return log_probs, entropy, state_values
