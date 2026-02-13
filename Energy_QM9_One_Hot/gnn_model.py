import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_geometric.nn import AttentionalAggregation
from torch_geometric.data import Batch
from typing import Dict, Tuple, Optional


class LOHCGNN(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int = 0,              # kept for backward-compat; not used in A2/A3 path
        line_node_in_dim: Optional[int] = None,
        line_edge_in_dim: int = 0,         # kept for backward-compat; not used in A2/A3 path
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_output_features: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        if line_node_in_dim is None:
            raise ValueError("line_node_in_dim must be provided (expected: TOTAL_FEATURE_DIMENSION + NUM_BOND_FEATURES).")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Atom embedding (used later when forming atom states)
        self.atom_embed = nn.Linear(node_in_dim, hidden_dim)

        # [CHANGED A1] Directed-bond node embedding
        self.bond_in = nn.Linear(line_node_in_dim, hidden_dim)

        # [CHANGED A3] Chemprop-like bond update: (bond_in + aggregated_neighbor_bonds) -> MLP -> GRUCell
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # [CHANGED A2] Build atom hidden from (atom_embed, sum_incoming_bond_hidden)
        self.atom_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Readout
        self.atom_att_pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))

        self.pred_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_output_features),
        )

    @staticmethod
    def _compute_ptr_from_batch(batch: torch.Tensor) -> torch.Tensor:
        """Fallback for Batch.ptr when running in environments where ptr may not exist."""
        if batch.numel() == 0:
            return torch.zeros((1,), dtype=torch.long, device=batch.device)
        num_graphs = int(batch.max().item()) + 1
        counts = torch.bincount(batch, minlength=num_graphs)
        ptr = torch.zeros((num_graphs + 1,), dtype=torch.long, device=batch.device)
        ptr[1:] = torch.cumsum(counts, dim=0)
        return ptr

    @staticmethod
    def _index_add_2d(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """out[index[i]] += src[i] for 2D tensors; wrapper to keep code readable."""
        if index.numel() == 0 or src.numel() == 0:
            return out
        out.index_add_(0, index, src)
        return out

    def forward(self, atom_data: Batch, line_data: Batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Atom base embeddings
        h_atom0 = self.atom_embed(atom_data.x)  # (N_atom, hidden)

        # Directed-bond node embeddings
        bond_input = F.relu(self.bond_in(line_data.x))  # (N_bond, hidden)
        h_bond = bond_input

        # [CHANGED A3] Bond message passing over directed-bond adjacency (no-backtracking already encoded in line_data.edge_index)
        # edge_index: [2, E_line], edges are i -> j where i=(k->u), j=(u->v), with backtracking excluded
        for _ in range(self.num_layers):
            if line_data.edge_index.numel() == 0 or h_bond.numel() == 0:
                m = torch.zeros_like(h_bond)
            else:
                src, dst = line_data.edge_index[0], line_data.edge_index[1]
                m = torch.zeros_like(h_bond)  # aggregate to each bond node (dst)
                self._index_add_2d(m, dst, h_bond[src])

            # Chemprop-like update input: concat(bond_input, aggregated_neighbor_messages)
            u = self.msg_mlp(torch.cat([bond_input, m], dim=-1))  # (N_bond, hidden)
            h_bond = self.gru(u, h_bond)  # (N_bond, hidden)
            h_bond = F.dropout(h_bond, p=self.dropout_rate, training=self.training)

        # [CHANGED A2] Aggregate directed-bond hidden -> destination atoms
        # Requires line_data.dst (local atom indices per molecule) produced by data_processing.py
        if not hasattr(line_data, 'dst'):
            raise AttributeError("line_data.dst is required for A2/A3 (store local dst atom index for each directed bond in data_processing.py).")

        if hasattr(atom_data, 'ptr') and atom_data.ptr is not None:
            atom_ptr = atom_data.ptr
        else:
            atom_ptr = self._compute_ptr_from_batch(atom_data.batch)

        # Compute global destination atom indices for each directed bond
        bond_graph_id = line_data.batch  # (N_bond,)
        atom_offset = atom_ptr[bond_graph_id]  # (N_bond,)
        global_dst = line_data.dst.to(atom_offset.device) + atom_offset  # (N_bond,)

        atom_msg = torch.zeros((atom_data.num_nodes, self.hidden_dim), device=h_atom0.device, dtype=h_atom0.dtype)
        if h_bond.numel() != 0:
            self._index_add_2d(atom_msg, global_dst, h_bond)

        h_atom = self.atom_mlp(torch.cat([h_atom0, atom_msg], dim=-1))  # (N_atom, hidden)

        # Attention weights (optional for inspection)
        att_logits = self.atom_att_pool.gate_nn(h_atom)
        alpha = softmax(att_logits, atom_data.batch)

        # Readout (graph-level)
        h_pool = self.atom_att_pool(h_atom, atom_data.batch)
        pred = self.pred_mlp(h_pool)

        return pred, {"atom_attention": alpha}
