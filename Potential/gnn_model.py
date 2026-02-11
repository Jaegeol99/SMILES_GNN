import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, AttentionalAggregation
from typing import Tuple

class EdgeGatedConv(MessagePassing):
    def __init__(self, node_in_dim: int, edge_in_dim: int, out_dim: int):
        super().__init__(aggr='add')
        self.node_mlp = nn.Linear(node_in_dim + edge_in_dim, out_dim)
        self.edge_mlp = nn.Linear(node_in_dim + node_in_dim + edge_in_dim, out_dim)
        self.gate_mlp = nn.Linear(node_in_dim + edge_in_dim, out_dim)

    def forward(self, x, edge_index, edge_attr) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        gate = torch.sigmoid(self.gate_mlp(torch.cat([x_i, edge_attr], dim=-1)))
        return gate * self.node_mlp(torch.cat([x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x, edge_index, edge_attr):
        row, col = edge_index
        new_edge_attr = self.edge_mlp(torch.cat([x[row], x[col], edge_attr], dim=-1))
        return aggr_out, new_edge_attr

class LOHCGNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, line_edge_in_dim, hidden_dim, num_layers, num_output_features, dropout_rate=0.0):
        super().__init__()
        # [핵심] 단일 인코더 구조 (가중치 공유)
        # Atom 인코더 1세트, Line 인코더 1세트 = 총 2세트
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim)
        self.line_edge_embed = nn.Linear(line_edge_in_dim, hidden_dim)

        self.atom_convs = nn.ModuleList([EdgeGatedConv(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.line_convs = nn.ModuleList([EdgeGatedConv(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)])

        # 풀링 레이어도 공유하여 동일한 기준으로 특징 추출
        self.pool_atom = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))
        self.pool_line = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))

        # 최종 예측 MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_output_features)
        )

    def forward(self, atom_data, line_data):
        # 1. Dehydrogenated 상태 처리 (Shared Encoder 사용)
        h_d = self.node_embed(atom_data.x_de)
        e_d = self.edge_embed(atom_data.edge_attr_de)
        l_d = self.edge_embed(line_data.x_de)
        le_d = self.line_edge_embed(line_data.edge_attr_de)

        for a_conv, l_conv in zip(self.atom_convs, self.line_convs):
            h_d_up, e_d_up = a_conv(h_d, atom_data.edge_index_de, e_d)
            l_d_up, le_d_up = l_conv(l_d, line_data.edge_index_de, le_d)
            h_d, e_d, l_d, le_d = h_d + h_d_up, e_d + e_d_up, l_d + l_d_up, le_d + le_d_up

        # 2. Hydrogenated 상태 처리 (동일한 Shared Encoder 사용)
        h_h = self.node_embed(atom_data.x)
        e_h = self.edge_embed(atom_data.edge_attr)
        l_h = self.edge_embed(line_data.x)
        le_h = self.line_edge_embed(line_data.edge_attr)

        for a_conv, l_conv in zip(self.atom_convs, self.line_convs):
            h_h_up, e_h_up = a_conv(h_h, atom_data.edge_index, e_h)
            l_h_up, le_h_up = l_conv(l_h, line_data.edge_index, le_h)
            h_h, e_h, l_h, le_h = h_h + h_h_up, e_h + e_h_up, l_h + l_h_up, le_h + le_h_up

        # 3. Readout (Pooling)
        p_hd = self.pool_atom(h_d, atom_data.batch_de)
        p_ld = self.pool_line(l_d, line_data.batch_de)
        p_hh = self.pool_atom(h_h, atom_data.batch)
        p_lh = self.pool_line(l_h, line_data.batch)

        # 4. 결합 및 예측
        return self.mlp(torch.cat([p_hd, p_ld, p_hh, p_lh], dim=-1))