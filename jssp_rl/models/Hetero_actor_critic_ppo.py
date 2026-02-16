import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from models.gin import HeteroGATv2

class ActorCriticPPO(nn.Module):
    def __init__(self, node_input_dim, gnn_hidden_dim, gnn_output_dim,
                 actor_hidden_dim, critic_hidden_dim):
        super(ActorCriticPPO, self).__init__()

        self.gnn = HeteroGATv2(
            in_channels=node_input_dim,
            hidden_dim=gnn_hidden_dim,
            out_channels=gnn_output_dim
        )

        self.actor_mlp = nn.Sequential(
            nn.Linear(gnn_output_dim, actor_hidden_dim),
            nn.GELU(),
            nn.Linear(actor_hidden_dim, 1)
        )

        self.critic_mlp = nn.Sequential(
            nn.Linear(gnn_output_dim, critic_hidden_dim),
            nn.GELU(),
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

    def act(self, data,  use_est_boost, est_beta, mask=None, env=None):
        action_logits, state_values = self.forward(data)

        if mask is not None:
            assert mask.dtype == torch.bool and mask.ndim == 1, "[Mask] Must be 1D bool"
            assert action_logits.shape[0] == mask.shape[0], \
                f"[MaskLen] logits:{action_logits.shape[0]} vs mask:{mask.shape[0]}"
            action_logits = action_logits.masked_fill(~mask, -1e10)
            

        # --- Apply heuristic boost if enabled ---
        if use_est_boost and env is not None:
            est_boost = env.get_est_boost(mask, est_beta=est_beta, device=action_logits.device )
            #print(f"[DBG] est_boost mean={est_boost[mask].float().mean().item():.4f} "
                   # f"max={est_boost[mask].float().max().item():.4f} "
                    #f"min={est_boost[mask].float().min().item():.4f}")
            
            valid_logits = action_logits[mask]

           # print(f"[DBG] valid logits before boost: mean={valid_logits.mean():.3f}, "
                  #  f"max={valid_logits.max():.3f}, min={valid_logits.min():.3f}")

            action_logits = action_logits + est_boost

            # valid_logits_after = action_logits[mask]
        
        

        action_dist = torch.distributions.Categorical(logits=action_logits)

        if self.training:
            action = action_dist.sample()
        else:
            action = torch.argmax(action_logits)

        log_prob = action_dist.log_prob(action)
        value = state_values.squeeze(-1)

        return action.item(), log_prob, value


    def evaluate_actions(self, data, actions, mask=None, use_est_boost=False, est_beta=1.0, env=None):
        action_logits, state_values = self.forward(data)

        if mask is not None:
            action_logits = action_logits.masked_fill(~mask, -1e10)

        if use_est_boost and env is not None:
            est_boost = env.get_est_boost(mask, est_beta=est_beta, device=action_logits.device)
            action_logits = action_logits + est_boost

        state_values = state_values.squeeze(-1)

        action_dist = torch.distributions.Categorical(logits=action_logits)

        if isinstance(actions, list):
            actions = torch.tensor(actions, dtype=torch.long, device=action_logits.device)
        else:
            actions = actions.to(dtype=torch.long)

        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()

        return log_probs, entropy, state_values

