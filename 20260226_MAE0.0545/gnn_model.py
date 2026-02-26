import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
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
        global_in_dim: int = 0,
    ):
        super().__init__()

        if line_node_in_dim is None:
            raise ValueError("line_node_in_dim must be provided (expected: TOTAL_FEATURE_DIMENSION + NUM_BOND_FEATURES).")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.global_in_dim = int(global_in_dim)

        # Atom embedding (used later when forming atom states)
        self.atom_embed = nn.Linear(node_in_dim, hidden_dim)

        # Directed-bond node embedding
        self.bond_in = nn.Linear(line_node_in_dim, hidden_dim)

        # D-MPNN 논문 원본 방식 (Residual Connection) 적용을 위한 선형 레이어
        self.W_m = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Build atom hidden from (atom_embed, sum_incoming_bond_hidden)
        self.atom_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # [수정됨] AttentionalAggregation 제거 (Sum Pooling 사용)

        # Global feature projection (optional)
        if self.global_in_dim > 0:
            self.global_proj = nn.Sequential(
                nn.Linear(self.global_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
            pred_in_dim = hidden_dim * 2  # [h_pool, global_emb]
        else:
            self.global_proj = None
            pred_in_dim = hidden_dim

        self.pred_mlp = nn.Sequential(
            nn.Linear(pred_in_dim, hidden_dim * 2),
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

        # Bond message passing over directed-bond adjacency
        for _ in range(self.num_layers):
            if line_data.edge_index.numel() == 0 or h_bond.numel() == 0:
                m = torch.zeros_like(h_bond)
            else:
                src, dst = line_data.edge_index[0], line_data.edge_index[1]
                m = torch.zeros_like(h_bond)  # aggregate to each bond node (dst)
                self._index_add_2d(m, dst, h_bond[src])

            # D-MPNN Update Rule: h_vw^(t+1) = ReLU(h_vw^(0) + W_m * m_vw)
            h_bond = F.relu(bond_input + self.W_m(m))  # (N_bond, hidden)
            h_bond = F.dropout(h_bond, p=self.dropout_rate, training=self.training)

        # Aggregate directed-bond hidden -> destination atoms
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

        # [수정됨] Readout (graph-level) using Sum Pooling (global_add_pool)
        h_pool = global_add_pool(h_atom, atom_data.batch)

        if self.global_proj is not None:
            if hasattr(atom_data, "g") and atom_data.g is not None:
                g = atom_data.g
            else:
                # Fallback: allow running even if global features are missing
                g = torch.zeros((h_pool.size(0), self.global_in_dim), device=h_pool.device, dtype=h_pool.dtype)
            g_emb = self.global_proj(g)
            h_pool = torch.cat([h_pool, g_emb], dim=-1)

        pred = self.pred_mlp(h_pool)

        # 어텐션 가중치가 없으므로 빈 dict 반환
        return pred, {}