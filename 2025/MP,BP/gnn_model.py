import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Batch
from typing import Tuple

class EdgeGatedConv(MessagePassing):
    def __init__(self, node_in_dim: int, edge_in_dim: int, out_dim: int):
        super().__init__(aggr='add')
        self.node_mlp = nn.Linear(node_in_dim + edge_in_dim, out_dim)
        self.edge_mlp = nn.Linear(node_in_dim + node_in_dim + edge_in_dim, out_dim)
        self.gate_mlp = nn.Linear(node_in_dim + edge_in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_mlp(torch.cat([x_i, edge_attr], dim=-1)))
        return gate * self.node_mlp(torch.cat([x_j, edge_attr], dim=-1))

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        new_x = aggr_out
        row, col = edge_index
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        new_edge_attr = self.edge_mlp(edge_input)
        return new_x, new_edge_attr

class LOHCGNN(nn.Module):
    def __init__(self, node_in_dim: int, edge_in_dim: int, line_edge_in_dim: int,
                 hidden_dim: int, num_layers: int, num_output_features: int, dropout_rate: float = 0.5):
        super().__init__()
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim)
        self.line_edge_embed = nn.Linear(line_edge_in_dim, hidden_dim)

        self.atom_conv_layers = nn.ModuleList([
            EdgeGatedConv(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.line_conv_layers = nn.ModuleList([
            EdgeGatedConv(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_output_features)
        )

    def forward(self, atom_data: Batch, line_data: Batch) -> torch.Tensor:
        h = self.node_embed(atom_data.x)
        e = self.edge_embed(atom_data.edge_attr)
        l = self.edge_embed(line_data.x)
        le = self.line_edge_embed(line_data.edge_attr)

        for atom_conv, line_conv in zip(self.atom_conv_layers, self.line_conv_layers):
            l_update, le_update = line_conv(l, line_data.edge_index, le)
            h_update, e_update = atom_conv(h, atom_data.edge_index, e)

            h = h + h_update
            e = e + e_update
            l = l + l_update
            le = le + le_update
            
        h_pooled = global_mean_pool(h, atom_data.batch)
        
        return self.mlp(h_pooled)