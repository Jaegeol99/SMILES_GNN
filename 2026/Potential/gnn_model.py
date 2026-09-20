# --- START OF FILE gnn_model.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, GCNConv, GINConv
from torch_geometric.data import Batch
from typing import Dict, Tuple, Optional

class LOHCGNN(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int = 0,              
        line_node_in_dim: Optional[int] = None,
        line_edge_in_dim: int = 0,         
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_output_features: int = 1,
        dropout_rate: float = 0.0,
        global_in_dim: int = 0,
        pooling_type: str = "mean",          # 기본값을 mean으로 변경
        use_line_edge_features: bool = True, 
    ):
        super().__init__()

        if line_node_in_dim is None:
            raise ValueError("line_node_in_dim must be provided.")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.global_in_dim = int(global_in_dim)
        self.pooling_type = pooling_type
        
        self.use_line_edge_features = use_line_edge_features

        # Atom embedding
        self.atom_embed = nn.Linear(node_in_dim, hidden_dim)

        # Directed-bond node embedding
        self.bond_in = nn.Linear(line_node_in_dim, hidden_dim)

        # 라인 엣지 피처(가상 각도, 고리 긴장 등)를 결합하여 메시지를 생성하는 신경망
        if self.use_line_edge_features and line_edge_in_dim > 0:
            self.line_edge_nn = nn.Sequential(
                nn.Linear(hidden_dim + line_edge_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        else:
            self.line_edge_nn = None

        # D-MPNN Update Rule
        self.W_m = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Atom hidden MLP
        self.atom_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.global_proj = None
        
        pred_in_dim = hidden_dim + self.global_in_dim

        mlp_hidden1 = pred_in_dim // 2
        mlp_hidden2 = mlp_hidden1 // 2
        mlp_hidden3 = mlp_hidden2 // 2
        mlp_hidden4 = mlp_hidden3 // 2

        self.pred_mlp = nn.Sequential(
            nn.Linear(pred_in_dim, mlp_hidden1),
            nn.GELU(),  
            nn.Dropout(dropout_rate),
            
            nn.Linear(mlp_hidden1, mlp_hidden2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(mlp_hidden2, mlp_hidden3),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(mlp_hidden3, mlp_hidden4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(mlp_hidden4, num_output_features)
        )

    @staticmethod
    def _compute_ptr_from_batch(batch: torch.Tensor) -> torch.Tensor:
        if batch.numel() == 0:
            return torch.zeros((1,), dtype=torch.long, device=batch.device)
        num_graphs = int(batch.max().item()) + 1
        counts = torch.bincount(batch, minlength=num_graphs)
        ptr = torch.zeros((num_graphs + 1,), dtype=torch.long, device=batch.device)
        ptr[1:] = torch.cumsum(counts, dim=0)
        return ptr

    @staticmethod
    def _index_add_2d(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        if index.numel() == 0 or src.numel() == 0:
            return out
        out.index_add_(0, index, src)
        return out

    def forward(self, atom_data: Batch, line_data: Batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h_atom0 = self.atom_embed(atom_data.x)  
        
        # 라플라시안 PE가 제거되었으므로 슬라이싱 없이 바로 사용
        bond_input = F.relu(self.bond_in(line_data.x))  
        h_bond = bond_input

        for _ in range(self.num_layers):
            if line_data.edge_index.numel() == 0 or h_bond.numel() == 0:
                m = torch.zeros_like(h_bond)
            else:
                src, dst = line_data.edge_index[0], line_data.edge_index[1]
                m = torch.zeros_like(h_bond)
                
                if self.line_edge_nn is not None and hasattr(line_data, 'edge_attr') and line_data.edge_attr is not None:
                    edge_attr = line_data.edge_attr
                    msg = self.line_edge_nn(torch.cat([h_bond[src], edge_attr], dim=-1))
                    self._index_add_2d(m, dst, msg)
                else:
                    self._index_add_2d(m, dst, h_bond[src])

            h_bond = F.relu(bond_input + self.W_m(m))
            h_bond = F.dropout(h_bond, p=self.dropout_rate, training=self.training)

        if not hasattr(line_data, 'dst'):
            raise AttributeError("line_data.dst is required.")

        atom_ptr = atom_data.ptr if hasattr(atom_data, 'ptr') and atom_data.ptr is not None else self._compute_ptr_from_batch(atom_data.batch)
        bond_graph_id = line_data.batch
        atom_offset = atom_ptr[bond_graph_id]
        global_dst = line_data.dst.to(atom_offset.device) + atom_offset

        atom_msg = torch.zeros((atom_data.num_nodes, self.hidden_dim), device=h_atom0.device, dtype=h_atom0.dtype)
        if h_bond.numel() != 0:
            self._index_add_2d(atom_msg, global_dst, h_bond)

        h_atom = self.atom_mlp(torch.cat([h_atom0, atom_msg], dim=-1))

        if self.pooling_type == "mean":
            h_pool = global_mean_pool(h_atom, atom_data.batch)
        else:
            h_pool = global_add_pool(h_atom, atom_data.batch)

        if self.global_in_dim > 0:
            g = atom_data.g if hasattr(atom_data, "g") and atom_data.g is not None else torch.zeros((h_pool.size(0), self.global_in_dim), device=h_pool.device, dtype=h_pool.dtype)
            h_pool = torch.cat([h_pool, g], dim=-1)

        pred = self.pred_mlp(h_pool)

        return pred, {}


class BaselineGCN(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_output_features: int = 1,
        dropout_rate: float = 0.0,
        global_in_dim: int = 0,
        pooling_type: str = "mean",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.global_in_dim = int(global_in_dim)
        self.pooling_type = pooling_type

        self.node_embed = nn.Linear(node_in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        pred_in_dim = hidden_dim + self.global_in_dim
        mlp_hidden1 = pred_in_dim // 2
        mlp_hidden2 = mlp_hidden1 // 2
        mlp_hidden3 = mlp_hidden2 // 2
        mlp_hidden4 = mlp_hidden3 // 2

        self.pred_mlp = nn.Sequential(
            nn.Linear(pred_in_dim, mlp_hidden1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden1, mlp_hidden2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden2, mlp_hidden3),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden3, mlp_hidden4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden4, num_output_features)
        )

    def forward(self, atom_data: Batch, line_data: Batch = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x, edge_index, batch = atom_data.x, atom_data.edge_index, atom_data.batch

        x = self.node_embed(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        if self.pooling_type == "mean":
            h_pool = global_mean_pool(x, batch)
        else:
            h_pool = global_add_pool(x, batch)

        if self.global_in_dim > 0:
            g = atom_data.g if hasattr(atom_data, "g") and atom_data.g is not None else torch.zeros((h_pool.size(0), self.global_in_dim), device=h_pool.device, dtype=h_pool.dtype)
            h_pool = torch.cat([h_pool, g], dim=-1)

        pred = self.pred_mlp(h_pool)

        return pred, {}


class BaselineGIN(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_output_features: int = 1,
        dropout_rate: float = 0.0,
        global_in_dim: int = 0,
        pooling_type: str = "mean",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.global_in_dim = int(global_in_dim)
        self.pooling_type = pooling_type

        self.node_embed = nn.Linear(node_in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))

        pred_in_dim = hidden_dim + self.global_in_dim
        mlp_hidden1 = pred_in_dim // 2
        mlp_hidden2 = mlp_hidden1 // 2
        mlp_hidden3 = mlp_hidden2 // 2
        mlp_hidden4 = mlp_hidden3 // 2

        self.pred_mlp = nn.Sequential(
            nn.Linear(pred_in_dim, mlp_hidden1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden1, mlp_hidden2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden2, mlp_hidden3),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden3, mlp_hidden4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden4, num_output_features)
        )

    def forward(self, atom_data: Batch, line_data: Batch = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x, edge_index, batch = atom_data.x, atom_data.edge_index, atom_data.batch

        x = self.node_embed(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        if self.pooling_type == "mean":
            h_pool = global_mean_pool(x, batch)
        else:
            h_pool = global_add_pool(x, batch)

        if self.global_in_dim > 0:
            g = atom_data.g if hasattr(atom_data, "g") and atom_data.g is not None else torch.zeros((h_pool.size(0), self.global_in_dim), device=h_pool.device, dtype=h_pool.dtype)
            h_pool = torch.cat([h_pool, g], dim=-1)

        pred = self.pred_mlp(h_pool)

        return pred, {}