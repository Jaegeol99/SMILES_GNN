import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool
from torch_geometric.data import Batch

class GNNEncoder(nn.Module):
    def __init__(self, num_node_features: int, hidden_channels: int, num_layers: int = 3,
                 dropout_rate: float = 0.5, heads: int = 4, output_heads: int = 1):
        super().__init__()
        self.convs = nn.ModuleList()
        curr_dim = num_node_features

        self.convs.append(GATConv(curr_dim, hidden_channels, heads=heads, dropout=dropout_rate, concat=True))
        curr_dim = hidden_channels * heads

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(curr_dim, hidden_channels, heads=heads, dropout=dropout_rate, concat=True))
            curr_dim = hidden_channels * heads

        self.convs.append(GATConv(curr_dim, hidden_channels, heads=output_heads, dropout=dropout_rate, concat=False))
        self.out_dim = hidden_channels * output_heads

    def forward(self, data: Batch) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        return global_max_pool(x, batch)

class PairedLOHCGNN(nn.Module):
    def __init__(self, num_node_features: int, hidden_channels: int, num_output_features: int,
                 gnn_layers: int = 3, mlp_hidden_factor: int = 1, dropout_rate: float = 0.5,
                 gat_heads: int = 4, gat_output_heads: int = 1):
        super().__init__()
        self.encoder = GNNEncoder(num_node_features, hidden_channels, num_layers=gnn_layers,
                                  dropout_rate=dropout_rate, heads=gat_heads, output_heads=gat_output_heads)
        mlp_input_size = self.encoder.out_dim * 2
        mlp_hidden = hidden_channels * mlp_hidden_factor
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden, num_output_features)
        )

    def forward(self, data1: Batch, data2: Batch) -> torch.Tensor:
        emb1 = self.encoder(data1)
        emb2 = self.encoder(data2)
        combined = torch.cat([emb1, emb2], dim=-1)
        return self.mlp(combined)