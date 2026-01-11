import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, GraphNorm, GATConv, GINConv
from torch_geometric.data import HeteroData


class HeteroGATv2(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, out_channels=64, dropout=0.1, heads=4):
        super(HeteroGATv2, self).__init__()

        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # --- First GATv2 layer with GraphNorm and residual ---
        self.conv1 = HeteroConv({
            ('op', 'job', 'op'): GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False),
            ('op', 'machine', 'op'): GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False),
        }, aggr='sum')

        self.norm1 = GraphNorm(hidden_dim)
        self.residual1 = nn.Identity()

        # --- Second GATv2 layer with residual ---
        self.conv2 = HeteroConv({
            ('op', 'job', 'op'): GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False),
            ('op', 'machine', 'op'): GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False),
        }, aggr='sum')

        self.norm2 = GraphNorm(hidden_dim)
        self.residual2 = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_op = self.input_proj(x_dict['op'])

        # --- First Layer with Residual ---
        x_dict = {'op': x_op}
        h1 = self.conv1(x_dict, edge_index_dict)['op']
        h1 = self.norm1(h1, batch=torch.zeros(h1.size(0), dtype=torch.long, device=h1.device))
        h1 = F.gelu(h1)
        #sc1 h1 = F.relu(h1)
        h1 = self.dropout(h1)
        x1 = x_op + h1  # Residual

        # --- Second Layer with Residual ---
        x_dict = {'op': x1}
        h2 = self.conv2(x_dict, edge_index_dict)['op']
        h2 = self.norm2(h2, batch=torch.zeros(h2.size(0), dtype=torch.long, device=h2.device))
        h2 = F.gelu(h2)
        #sc1 h2 = F.relu(h2)
        h2 = self.dropout(h2)
        x2 = x1 + h2  # Residual

        return self.output_proj(x2)

    @staticmethod
    def build_hetero_data(x, job_edges, machine_edges):
        data = HeteroData()
        data['op'].x = x
        data['op', 'job', 'op'].edge_index = job_edges
        data['op', 'machine', 'op'].edge_index = machine_edges
        return data

    @staticmethod
    def build_edge_index_dict(machines: torch.Tensor):
        machines = machines.detach().cpu()
        num_jobs, num_machines = machines.shape

        job_edges = []
        for j in range(num_jobs):
            for o in range(num_machines - 1):
                k = j * num_machines + o
                kn = j * num_machines + (o + 1)
                job_edges.append([k, kn])
                job_edges.append([kn, k])
        job_edge_index = torch.tensor(job_edges, dtype=torch.long).t().contiguous() if job_edges else torch.empty(2, 0, dtype=torch.long)

        machine_to_ops = {}
        for j in range(num_jobs):
            for o in range(num_machines):
                k = j * num_machines + o
                m = int(machines[j, o].item())
                machine_to_ops.setdefault(m, []).append(k)

        machine_edges = []
        for ops in machine_to_ops.values():
            for i in range(len(ops)):
                for j in range(i + 1, len(ops)):
                    machine_edges.append([ops[i], ops[j]])
                    machine_edges.append([ops[j], ops[i]])
        machine_edge_index = torch.tensor(machine_edges, dtype=torch.long).t().contiguous() if machine_edges else torch.empty(2, 0, dtype=torch.long)

        return {
            ('op', 'job', 'op'): job_edge_index,
            ('op', 'machine', 'op'): machine_edge_index
        }
      
# sc4 Added HeteroGAT and HeteroGIN variants for experimentation; same interface as HeteroGATv2.
# Switch the class used in models/Hetero_actor_critic_ppo.py to compare GAT/GATv2/GIN backbones.
class HeteroGAT(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, out_channels=64, dropout=0.1, heads=4):
        super(HeteroGAT, self).__init__()

        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # --- First GAT layer with GraphNorm and residual ---
        self.conv1 = HeteroConv({
            ('op', 'job','op'): GATConv(hidden_dim, hidden_dim, heads=heads, concat=False),
            ('op', 'machine', 'op'): GATConv(hidden_dim, hidden_dim, heads=heads, concat=False),
        }, aggr='sum')

        self.norm1 = GraphNorm(hidden_dim)
        self.residual1 = nn.Identity()

        # --- Second GAT layer with residual ---
        self.conv2 = HeteroConv({
            ('op', 'job', 'op'): GATConv(hidden_dim, hidden_dim, heads=heads, concat=False),
            ('op', 'machine', 'op'): GATConv(hidden_dim, hidden_dim, heads=heads, concat=False),
        }, aggr='sum')

        self.norm2 = GraphNorm(hidden_dim)
        self.residual2 = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_op = self.input_proj(x_dict['op'])

        # --- First Layer with Residual ---
        x_dict = {'op': x_op}
        h1 = self.conv1(x_dict, edge_index_dict)['op']
        h1 = self.norm1(h1, batch=torch.zeros(h1.size(0), dtype=torch.long, device=h1.device))
        h1 = F.gelu(h1)
        h1 = self.dropout(h1)
        x1 = x_op + h1  # Residual

        # --- Second Layer with Residual ---
        x_dict = {'op': x1}
        h2 = self.conv2(x_dict, edge_index_dict)['op']
        h2 = self.norm2(h2, batch=torch.zeros(h2.size(0), dtype=torch.long, device=h2.device))
        h2 = F.gelu(h2)
        h2 = self.dropout(h2)
        x2 = x1 + h2  # Residual

        return self.output_proj(x2)

    @staticmethod
    def build_hetero_data(x, job_edges, machine_edges):
        return HeteroGATv2.build_hetero_data(x, job_edges, machine_edges)

    @staticmethod
    def build_edge_index_dict(machines: torch.Tensor):
        return HeteroGATv2.build_edge_index_dict(machines)


class HeteroGIN(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, out_channels=64, dropout=0.1):
        super(HeteroGIN, self).__init__()

        self.input_proj = nn.Linear(in_channels, hidden_dim)

        mlp1_job = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        mlp1_machine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- First GIN layer with GraphNorm and residual ---
        self.conv1 = HeteroConv({
            ('op', 'job', 'op'): GINConv(mlp1_job),
            ('op', 'machine', 'op'): GINConv(mlp1_machine),
        }, aggr='sum')

        self.norm1 = GraphNorm(hidden_dim)
        self.residual1 = nn.Identity()

        mlp2_job = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        mlp2_machine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- Second GIN layer with residual ---
        self.conv2 = HeteroConv({
            ('op', 'job', 'op'): GINConv(mlp2_job),
            ('op', 'machine', 'op'): GINConv(mlp2_machine),
        }, aggr='sum')

        self.norm2 = GraphNorm(hidden_dim)
        self.residual2 = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_op = self.input_proj(x_dict['op'])

        # --- First Layer with Residual ---
        x_dict = {'op': x_op}
        h1 = self.conv1(x_dict, edge_index_dict)['op']
        h1 = self.norm1(h1, batch=torch.zeros(h1.size(0), dtype=torch.long, device=h1.device))
        h1 = F.gelu(h1)
        h1 = self.dropout(h1)
        x1 = x_op + h1  # Residual

        # --- Second Layer with Residual ---
        x_dict = {'op': x1}
        h2 = self.conv2(x_dict, edge_index_dict)['op']
        h2 = self.norm2(h2, batch=torch.zeros(h2.size(0), dtype=torch.long, device=h2.device))
        h2 = F.gelu(h2)
        h2 = self.dropout(h2)
        x2 = x1 + h2  # Residual

        return self.output_proj(x2)

    @staticmethod
    def build_hetero_data(x, job_edges, machine_edges):
        return HeteroGATv2.build_hetero_data(x, job_edges, machine_edges)

    @staticmethod
    def build_edge_index_dict(machines: torch.Tensor):
        return HeteroGATv2.build_edge_index_dict(machines)

