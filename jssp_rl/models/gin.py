import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, HeteroConv
from torch.nn import LayerNorm
from torch_geometric.data import HeteroData

class HeteroGIN(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, out_channels=64, dropout=0.1):
        super(HeteroGIN, self).__init__()

        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # Separate GINConv layers for each edge type
        self.conv1 = HeteroConv({
            ('op', 'job', 'op'): GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            ),
            ('op', 'machine', 'op'): GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            )
        }, aggr='sum')

        self.norm1 = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict['op'] = self.input_proj(x_dict['op'])
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict['op'] = self.norm1(x_dict['op'])
        x_dict['op'] = F.gelu(x_dict['op'])
        x_dict['op'] = self.dropout(x_dict['op'])
        return self.output_proj(x_dict['op'])

    @staticmethod
    def build_hetero_data(x, job_edges, machine_edges):
        data = HeteroData()
        data['op'].x = x
        data['op', 'job', 'op'].edge_index = job_edges
        data['op', 'machine', 'op'].edge_index = machine_edges
        return data

    @staticmethod
    def build_edge_index_dict(machines):
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

        job_edge_index = torch.tensor(job_edges, dtype=torch.long).t().contiguous()
        machine_edge_index = torch.tensor(machine_edges, dtype=torch.long).t().contiguous()

        return {
            ('op', 'job', 'op'): job_edge_index,
            ('op', 'machine', 'op'): machine_edge_index
        }
