import torch
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing, AttentionalAggregation
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
    def __init__(self, node_in_dim: int, edge_in_dim: int, line_node_in_dim: int, line_edge_in_dim: int,
                 hidden_dim: int, num_layers: int, num_output_features: int, dropout_rate: float = 0.5):
        super().__init__()
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.line_node_embed = nn.Linear(line_node_in_dim, hidden_dim)
        self.line_edge_embed = nn.Linear(line_edge_in_dim, hidden_dim)
        
        self.line_conv_layers = nn.ModuleList([
            EdgeGatedConv(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.atom_att_pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_output_features)
        )

    def _get_ptr_from_batch(self, batch: torch.Tensor) -> torch.Tensor:
        # Fallback if atom_data.ptr is not available.
        # ptr[g] = starting node index of graph g in the concatenated batch.
        num_graphs = int(batch.max().item() + 1) if batch.numel() > 0 else 0
        counts = torch.bincount(batch, minlength=num_graphs)
        ptr = torch.zeros((num_graphs + 1,), device=batch.device, dtype=torch.long)
        ptr[1:] = torch.cumsum(counts, dim=0)
        return ptr
    
    def forward(self, atom_data: Batch, line_data: Batch) -> torch.Tensor:
        # Base atom states (used for readout and to support graphs with no bonds)
        h0 = self.node_embed(atom_data.x)

        # Directed-bond (line) graph states
        l = self.line_node_embed(line_data.x)
        le = self.line_edge_embed(line_data.edge_attr)

        for line_conv in self.line_conv_layers:
            l_upd, le_upd = line_conv(l, line_data.edge_index, le)
            l = l + l_upd
            le = le + le_upd

        # [CHANGED A2] Aggregate directed-bond states -> atom states (sum of incoming bonds)
        # line_data.dst stores *local* destination atom indices per molecule.
        if hasattr(atom_data, "ptr") and atom_data.ptr is not None:
            ptr = atom_data.ptr
        else:
            ptr = self._get_ptr_from_batch(atom_data.batch)

        if hasattr(line_data, "dst"):
            dst_local = line_data.dst
        else:
            # Backward compatible: if dst not provided, fall back to zeros (no bond contribution).
            dst_local = torch.zeros((l.size(0),), device=l.device, dtype=torch.long)

        if line_data.batch.numel() > 0:
            offsets = ptr[line_data.batch]  # start index of the corresponding molecule's atom block
            dst_global = dst_local.to(offsets.device) + offsets
        else:
            dst_global = dst_local

        atom_h = torch.zeros((atom_data.num_nodes, l.size(-1)), device=l.device, dtype=l.dtype)
        if dst_global.numel() > 0 and l.numel() > 0:
            atom_h.index_add_(0, dst_global, l)

        # Combine base atom features and message-aggregated bond info (common D-MPNN variant)
        h = h0 + atom_h

        # Attention pooling across atoms per graph
        att_logits = self.atom_att_pool.gate_nn(h)
        alpha = softmax(att_logits, atom_data.batch)
        h_att = self.atom_att_pool(h, atom_data.batch)

        pred = self.mlp(h_att)
        return pred, {"atom_attention": alpha}