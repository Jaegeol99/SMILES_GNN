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


        # Attention Pooling Layers
        self.atom_att_pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))
        self.line_att_pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))


        # combined = [atom_pool (hidden_dim), line_pool (hidden_dim)] -> hidden_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_output_features)
        )

    def forward(self, atom_data: Batch, line_data: Batch) -> torch.Tensor:
        h = self.node_embed(atom_data.x)
        e = self.edge_embed(atom_data.edge_attr)

        l = self.edge_embed(line_data.x)
        le = self.line_edge_embed(line_data.edge_attr)

        for atom_conv, line_conv in zip(self.atom_conv_layers, self.line_conv_layers):
            l_upd, le_upd = line_conv(l, line_data.edge_index, le)
            h_upd, e_upd = atom_conv(h, atom_data.edge_index, e)

            h = h + h_upd
            e = e + e_upd
            l = l + l_upd
            le = le + le_upd

        att_logits = self.atom_att_pool.gate_nn(h)
        alpha = softmax(att_logits, atom_data.batch)

        h_att = self.atom_att_pool(h,atom_data.batch)
        l_att = self.line_att_pool(l,line_data.batch)

        num_graphs = getattr(atom_data, 'num_graphs', None)
        if num_graphs is None:
            num_graphs = int(atom_data.batch.max().item() + 1) if atom_data.batch.numel() > 0 else 0

        # l_att의 크기가 num_graphs보다 작다면 (빈 그래프 존재 시) 패딩 처리
        if l_att.shape[0] != num_graphs:
            # 전체가 0인 텐서 생성
            l_att_full = torch.zeros((num_graphs, h_att.shape[1]), device=l_att.device, dtype=l_att.dtype)
            
            if l_att.numel() > 0:
                # 존재하는 그래프의 인덱스를 찾아 해당 위치에 값 할당
                present_idxs = torch.unique(line_data.batch)
                l_att_full[present_idxs] = l_att
        else:
            # 크기가 맞으면 그대로 사용
            l_att_full = l_att
        # ---------------------------------------------------------

        # [중요] 반드시 l_att 대신 l_att_full을 사용해야 합니다.
        combined = torch.cat([h_att, l_att_full], dim=-1)
        pred = self.mlp(combined)

        return pred, {"atom_attention": alpha}