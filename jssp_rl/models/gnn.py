import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import LayerNorm

class GNNWithAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, out_channels=64, num_heads=4, dropout=0.1):
        super(GNNWithAttention, self).__init__()
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        self.gat1 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True)

        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.output_proj = nn.Linear(hidden_dim, out_channels)
        
    @staticmethod
    def build_edge_index_with_machine_links(machines):
        """
        Args:
            machines: Tensor of shape [num_jobs, num_machines]
        Returns:
            edge_index: torch.LongTensor [2, num_edges]
        """
        num_jobs, num_machines = machines.shape
        job_edges = []
        machine_to_ops = {}

        for j in range(num_jobs):
            for o in range(num_machines):
                curr = j * num_machines + o
                if o < num_machines - 1:
                    nxt = j * num_machines + o + 1
                    job_edges.append([curr, nxt])

                machine = int(machines[j, o])
                machine_to_ops.setdefault(machine, []).append(curr)

        machine_edges = []
        for ops in machine_to_ops.values():
            for i in range(len(ops)):
                for j in range(i + 1, len(ops)):
                    machine_edges.append([ops[i], ops[j]])
                    machine_edges.append([ops[j], ops[i]])

        edge_index = torch.tensor(job_edges + machine_edges, dtype=torch.long).t().contiguous()
        return edge_index

    def forward(self, x, edge_index, edge_weights=None):
        x = self.input_proj(x)
        x = self.norm1(x)
        x = F.gelu(x)

        x = self.gat1(x, edge_index)
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.gat2(x, edge_index)
        x = self.dropout(x)

        return self.output_proj(x)
