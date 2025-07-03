
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
