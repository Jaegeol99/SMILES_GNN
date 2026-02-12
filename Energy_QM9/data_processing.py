import pandas as pd
import numpy as np
import logging

from rdkit import Chem
from rdkit.Chem import AllChem

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List, Optional, Tuple

from config import CSV_PATH, CSV_SMILES_COL, CSV_TARGET_COL, CSV_INDEX_COL, MAX_SAMPLES
from feature_configs import *

_feature_min_values = ATOM_FEATURE_MIN_VALUES_SAFE
_feature_max_values = ATOM_FEATURE_MAX_VALUES_SAFE

def smiles_to_graph_data(smiles: str, labels: list) -> Optional[Tuple[Data, Data]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0: return None
    
    try: AllChem.ComputeGasteigerCharges(mol)
    except: 
        for atom in mol.GetAtoms(): atom.SetDoubleProp('_GasteigerCharge', 0.0)
    
    func_group_matches = {gn: {idx for match in mol.GetSubstructMatches(pat) for idx in match} 
                          for gn, pat in FUNCTIONAL_GROUP_PATTERNS.items()}

    # 1. Atom Features
    atom_features = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        g_charge = float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0
        basic = [float(atom.GetAtomicNum()), float(atom.GetHybridization()), float(atom.GetIsAromatic()), 
                 float(atom.GetTotalNumHs()), float(atom.GetFormalCharge()), g_charge, float(atom.GetDegree()), float(atom.IsInRing())]
        neighbors = [0.0] * NUM_NEIGHBOR_FEATURES
        for n in atom.GetNeighbors():
            if n.GetAtomicNum() in NEIGHBOR_ATOMS_IDX: neighbors[NEIGHBOR_ATOMS_IDX[n.GetAtomicNum()]] += 1.0
        funcs = [1.0 if i in func_group_matches[gn] else 0.0 for gn in FUNCTIONAL_GROUP_PATTERNS.keys()]
        atom_features.append(basic + neighbors + funcs)
    
    x_atom = torch.tensor(atom_features, dtype=torch.float)
    if x_atom.shape[1] != TOTAL_FEATURE_DIMENSION:
        raise ValueError(f"Atom feature dim mismatch: got {x_atom.shape[1]}, expected {TOTAL_FEATURE_DIMENSION}")
    x_scaled = scale_atom_features_safe(x_atom)

    # 2. Directed Edges for D-MPNN (Line Graph Nodes)
    directed_edges = []
    edge_features = []
    atom_edge_index = []
    atom_edge_attr = []

    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        bond_feat = [float(bt == Chem.rdchem.BondType.SINGLE), float(bt == Chem.rdchem.BondType.DOUBLE),
                     float(bt == Chem.rdchem.BondType.TRIPLE), float(bt == Chem.rdchem.BondType.AROMATIC),
                     float(bond.GetIsConjugated()), float(bond.IsInRing())]
        
        # Atom Graph (Standard)
        atom_edge_index.extend([[u, v], [v, u]])
        atom_edge_attr.extend([bond_feat, bond_feat])
        
        # Line Graph Nodes (Directed Edges)
        directed_edges.append((u, v))
        edge_features.append(bond_feat)
        directed_edges.append((v, u))
        edge_features.append(bond_feat)

    # Build line-node feature tensor. If there are no bonds, create an empty
    # tensor with shape (0, NUM_BOND_FEATURES) to avoid indexing shape[1]
    if len(edge_features) == 0:
        x_line = torch.empty((0, NUM_BOND_FEATURES), dtype=torch.float)
    else:
        x_line = torch.tensor(edge_features, dtype=torch.float)
        # If torch created a 1-D tensor (unexpected), reshape to (N, F)
        if x_line.dim() == 1:
            x_line = x_line.unsqueeze(0)
        if x_line.shape[1] != NUM_BOND_FEATURES:
            raise ValueError(f"Bond feature dim mismatch: got {x_line.shape[1]}, expected {NUM_BOND_FEATURES}")

    # Atom graph edge_index / edge_attr: ensure well-formed empty tensors when
    # there are no bonds
    if len(atom_edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(atom_edge_index, dtype=torch.long).t().contiguous()

    if len(atom_edge_attr) == 0:
        edge_attr = torch.empty((0, NUM_BOND_FEATURES), dtype=torch.float)
    else:
        edge_attr = torch.tensor(atom_edge_attr, dtype=torch.float)

    # 3. Directed Line Graph Edges (No-backtracking)
    line_edge_indices, line_edge_attrs = [], []
    num_directed = len(directed_edges)
    for i in range(num_directed):
        u_i, v_i = directed_edges[i]
        for j in range(num_directed):
            if i == j: continue
            u_j, v_j = directed_edges[j]
            
            # No-backtracking: i의 끝이 j의 시작이고, j의 끝이 i의 시작이 아닐 때만 연결
            if v_i == u_j and u_i != v_j:
                atom = mol.GetAtomWithIdx(v_i)
                hyb = atom.GetHybridization()
                hyb_feat = [float(hyb == Chem.rdchem.HybridizationType.SP), 
                            float(hyb == Chem.rdchem.HybridizationType.SP2), 
                            float(hyb == Chem.rdchem.HybridizationType.SP3)]
                le_feat = hyb_feat + edge_features[i] + edge_features[j]
                line_edge_indices.append([i, j])
                line_edge_attrs.append(le_feat)

    line_edge_index = torch.tensor(line_edge_indices, dtype=torch.long).t().contiguous() if line_edge_indices else torch.empty((2, 0), dtype=torch.long)
    line_edge_attr = torch.tensor(line_edge_attrs, dtype=torch.float) if line_edge_attrs else torch.empty((0, NUM_LINE_EDGE_FEATURES), dtype=torch.float)
    if line_edge_attr.numel() != 0 and line_edge_attr.shape[1] != NUM_LINE_EDGE_FEATURES:
        raise ValueError(f"Line-edge feature dim mismatch: got {line_edge_attr.shape[1]}, expected {NUM_LINE_EDGE_FEATURES}")

    y = torch.tensor([labels], dtype=torch.float)
    return Data(x=x_scaled, edge_index=edge_index, edge_attr=edge_attr, y=y), \
           Data(x=x_line, edge_index=line_edge_index, edge_attr=line_edge_attr, y=y)

def load_and_preprocess_qm9_data():
    """
    QM9에서 SMILES와 타깃(에너지)을 가져와,
    기존 smiles_to_graph_data()로 (atom_graph, line_graph)를 생성.
    """
    path = CSV_PATH
    df = pd.read_csv(path)
    data_list = []

    if MAX_SAMPLES is not None:
        df = df.iloc[:MAX_SAMPLES].copy()

    for _, row in df.iterrows():
        smiles = row.get(CSV_SMILES_COL, None)
        target = row.get(CSV_TARGET_COL, None)


        if not isinstance(smiles, str) or len(smiles) == 0:
            continue
        if target is None or (isinstance(target, float) and np.isnan(target)):
            continue

        res = smiles_to_graph_data(smiles, [float(target)])
        if res is not None:
            data_list.append(res)

    return data_list, TOTAL_FEATURE_DIMENSION


def calculate_label_scaling_params(data_list, train_indices):
    train_labels = np.concatenate([data_list[i][0].y.numpy() for i in train_indices], axis=0)  # (N, 1)
    median = np.nanmedian(train_labels, axis=0)
    q1, q3 = np.nanpercentile(train_labels, [25, 75], axis=0)
    iqr = np.where((q3 - q1) < 1e-9, 1.0, q3 - q1)
    return median, iqr

def apply_label_scaling(data_list, median, iqr):
    for atom_g, line_g in data_list:
        scaled_y = torch.tensor((atom_g.y.numpy() - median) / iqr, dtype=torch.float)
        atom_g.y = scaled_y
        line_g.y = scaled_y

def inverse_scale_labels(scaled, median, iqr):
    return scaled * iqr + median

def create_dataloaders(all_data, train_idx, val_idx, test_idx, batch_size):
    return (
        DataLoader([all_data[i] for i in train_idx], batch_size=batch_size, shuffle=True),
        DataLoader([all_data[i] for i in val_idx], batch_size=batch_size),
        DataLoader([all_data[i] for i in test_idx], batch_size=batch_size),
    )
