# --- START OF FILE gnn_model.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
# Using GATConv and global_max_pool as per the latest config/request
from torch_geometric.nn import GATConv, global_max_pool
from torch_geometric.data import Data, Batch # Batch is already imported
from typing import Optional

class GNNEncoder(nn.Module):
    """
    Graph Encoder using multiple GAT layers followed by global max pooling.
    """
    def __init__(self, num_node_features: int, hidden_channels: int, num_layers: int = 3,
                 dropout_rate: float = 0.5, heads: int = 4, output_heads: int = 1):
        super().__init__()
        if num_layers < 1: raise ValueError("GNNEncoder must have at least one layer.")

        self.convs = nn.ModuleList()
        current_dim = num_node_features

        # Input layer
        # Note: GATConv input dim is num_node_features, output is hidden_channels * heads (if concat=True)
        self.convs.append(GATConv(current_dim, hidden_channels, heads=heads, dropout=dropout_rate, concat=True))
        current_dim = hidden_channels * heads # Update dimension for next layer

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(current_dim, hidden_channels, heads=heads, dropout=dropout_rate, concat=True))
            current_dim = hidden_channels * heads

        # Output layer
        # Final GAT layer often uses concat=False and output_heads=1 (or matches MLP input needs)
        # Output dim will be hidden_channels * output_heads
        self.convs.append(GATConv(current_dim, hidden_channels, heads=output_heads, dropout=dropout_rate, concat=False))
        self.output_channels = hidden_channels * output_heads # Final embedding dimension per graph

        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self._reset_parameters() # Initialize parameters

    def _reset_parameters(self):
        """Initialize parameters of GAT layers."""
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data: Batch) -> torch.Tensor:
        """ Process graph batch data through GAT layers and global max pooling. """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Basic validation for input data
        if x is None or edge_index is None or batch is None:
            raise ValueError("GNNEncoder input data (x, edge_index, or batch) is missing.")
        # Handle empty graphs within a batch if necessary
        if x.numel() == 0:
            # Return zeros of the correct shape for the number of graphs in the batch
            num_graphs = data.num_graphs if hasattr(data, 'num_graphs') else batch.max().item() + 1
            return torch.zeros((num_graphs, self.output_channels), device=x.device)

        # Apply GAT layers with activation and dropout
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # Apply activation (e.g., ELU or ReLU) except for the last layer
            if i < self.num_layers - 1:
                x = F.elu(x) # ELU is common with GAT
                # Optional: Add dropout between layers if not handled by GATConv dropout
                # x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Apply global pooling (Max pooling in this case)
        # global_max_pool aggregates node features into a single graph embedding
        graph_embedding = global_max_pool(x, batch) # Input: node features, batch index

        return graph_embedding

class PairedLOHCGNN(nn.Module):
    """
    Paired LOHC GNN using a shared GAT Encoder (with global max pooling)
    and an MLP head for final prediction.
    """
    def __init__(self, num_node_features: int, hidden_channels: int, num_output_features: int,
                 gnn_layers: int = 3, mlp_hidden_factor: int = 1, dropout_rate: float = 0.5,
                 gat_heads: int = 4, gat_output_heads: int = 1):
        super().__init__()

        # Shared GNN Encoder instance
        self.gnn_encoder = GNNEncoder(
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            num_layers=gnn_layers,
            dropout_rate=dropout_rate,
            heads=gat_heads,
            output_heads=gat_output_heads
        )

        # MLP head to process concatenated graph embeddings
        # Input size is twice the encoder output size (one for dehydro, one for hydro)
        mlp_input_size = self.gnn_encoder.output_channels * 2
        # Hidden size can be based on hidden_channels or a factor
        mlp_hidden_size = hidden_channels * mlp_hidden_factor # Adjust factor as needed

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, mlp_hidden_size),
            nn.ReLU(), # Activation function
            nn.Dropout(p=dropout_rate), # Dropout for regularization
            nn.Linear(mlp_hidden_size, num_output_features) # Final output layer
        )

        self._reset_mlp_parameters() # Initialize MLP parameters

    def _reset_mlp_parameters(self):
        """Initialize parameters of the MLP layers."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                # Xavier initialization is common for linear layers with ReLU
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias) # Initialize bias to zero

    def forward(self, data_dehydro: Batch, data_hydro: Batch) -> torch.Tensor:
        """
        Forward pass for the paired GNN.
        Encodes both graphs, concatenates embeddings, and passes through MLP.
        """
        # Encode both dehydrogenated and hydrogenated graphs using the shared encoder
        embedding_dehydro = self.gnn_encoder(data_dehydro)
        embedding_hydro = self.gnn_encoder(data_hydro)

        # Concatenate the embeddings along the feature dimension
        combined_embedding = torch.cat([embedding_dehydro, embedding_hydro], dim=-1)

        # Pass the combined embedding through the MLP to get the final prediction
        output = self.mlp(combined_embedding)

        return output

# --- END OF FILE gnn_model.py ---