# gnn_model.py
"""
Define Graph Neural Network encoder and paired LOHC GNN model for multi-target energy prediction.
Uses GATConv layers followed by global max pooling and MLP head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool


class GNNEncoder(nn.Module):
    """
    Graph encoder using multiple GATConv layers and global max pooling.
    """
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int,
        num_layers: int = 3,
        dropout_rate: float = 0.5,
        gat_heads: int = 4,
        gat_output_heads: int = 1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("GNNEncoder requires at least one layer.")

        self.convs = nn.ModuleList()
        in_dim = num_node_features

        # Input GAT layer
        self.convs.append(
            GATConv(
                in_dim,
                hidden_channels,
                heads=gat_heads,
                dropout=dropout_rate,
                concat=True,
            )
        )
        in_dim = hidden_channels * gat_heads

        # Hidden GAT layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    in_dim,
                    hidden_channels,
                    heads=gat_heads,
                    dropout=dropout_rate,
                    concat=True,
                )
            )
            in_dim = hidden_channels * gat_heads

        # Output GAT layer
        self.convs.append(
            GATConv(
                in_dim,
                hidden_channels,
                heads=gat_output_heads,
                dropout=dropout_rate,
                concat=False,
            )
        )
        self.output_dim = hidden_channels * gat_output_heads
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize all GATConv layer parameters."""
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        """
        Forward pass through GAT layers and global max pooling.

        Args:
            data: Batch object with x, edge_index, batch attributes.
        Returns:
            Tensor of shape [num_graphs, output_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if x is None or edge_index is None or batch is None:
            raise ValueError("Missing input attributes in Batch: x, edge_index, or batch.")

        # Apply GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Pool node embeddings to graph level
        pooled = global_max_pool(x, batch)  # shape [num_graphs, output_dim]
        return pooled


class PairedLOHCGNN(nn.Module):
    """
    Siamese GNN model for paired dehydrogenated and hydrogenated SMILES graphs.
    Combines encoded representations with MLP head for multi-target regression.
    """
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int,
        num_output_features: int,
        dropout_rate: float = 0.5,
        gnn_layers: int = 3,
        gat_heads: int = 4,
        gat_output_heads: int = 1,
    ):
        super().__init__()
        # Shared graph encoder
        self.encoder = GNNEncoder(
            num_node_features,
            hidden_channels,
            num_layers=gnn_layers,
            dropout_rate=dropout_rate,
            gat_heads=gat_heads,
            gat_output_heads=gat_output_heads,
        )

        combined_dim = self.encoder.output_dim * 2
        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_channels, num_output_features),
        )

    def forward(self, batch):
        """
        Forward pass for paired graphs.

        Args:
            batch: Batch object containing 2*N graphs in sequence: [de1, hy1, de2, hy2, ...]
        Returns:
            Tensor of shape [N, num_output_features]
        """
        # Separate paired batches
        num_graphs = batch.num_graphs // 2
        embeddings = self.encoder(batch)

        # Split into dehydro and hydro embeddings
        de_embed = embeddings[0::2]
        hy_embed = embeddings[1::2]

        # Concatenate pairs
        pair_embed = torch.cat([de_embed, hy_embed], dim=1)
        # Predict targets
        out = self.mlp(pair_embed)
        return out
