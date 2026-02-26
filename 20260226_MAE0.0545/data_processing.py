from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from config import CSV_INDEX_COL, CSV_PATH, CSV_SMILES_COL, CSV_TARGET_COL, MAX_SAMPLES
from feature_configs import (
    FUNCTIONAL_GROUP_PATTERNS,
    LINE_NODE_FEATURE_DIM,
    MAX_GASTEIGER_ABS,
    MAX_NEIGHBOR_COUNT,
    NEIGHBOR_ATOMS_IDX,
    NUM_BOND_FEATURES,
    NUM_LINE_EDGE_FEATURES,
    NUM_NEIGHBOR_FEATURES,
    TOTAL_FEATURE_DIMENSION,
    one_hot_atom_type,
    one_hot_degree,
    one_hot_formal_charge,
    one_hot_hybridization,
    one_hot_total_h,
    scale_count,
    scale_signed,
    GLOBAL_FEATURE_DIM,
)


def compute_global_features(mol: Chem.Mol) -> torch.Tensor:
    """Compute graph-level (molecule-level) features from an RDKit Mol.
    [수정됨] RDKit이 제공하는 200여개의 모든 2D Descriptor를 일괄 계산합니다.
    """
    try:
        desc_list =[]
        for name, func in Descriptors.descList:
            try:
                val = float(func(mol))
            except Exception:
                val = 0.0
            desc_list.append(val)
        desc = np.array(desc_list, dtype=np.float32)
    except Exception:
        desc = np.zeros((GLOBAL_FEATURE_DIM,), dtype=np.float32)

    desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    if desc.shape[0] != GLOBAL_FEATURE_DIM:
        raise ValueError(f"Global feature dim mismatch: got {desc.shape[0]}, expected {GLOBAL_FEATURE_DIM}")

    return torch.from_numpy(desc).view(1, -1)


def smiles_to_graph_data(smiles: str, labels: List[float], mol_id: Optional[int] = None) -> Optional[Tuple[Data, Data]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None

    # Charges (may fail for some molecules)
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        for atom in mol.GetAtoms():
            atom.SetDoubleProp("_GasteigerCharge", 0.0)

    # Functional-group membership (multi-hot)
    func_group_matches = {
        name: {idx for match in mol.GetSubstructMatches(pat) for idx in match}
        for name, pat in FUNCTIONAL_GROUP_PATTERNS.items()
    }

    # -------------------------
    # Atom features (mixed: one-hot categorical + continuous)
    # -------------------------
    atom_features: List[List[float]] =[]
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        g_charge = float(atom.GetProp("_GasteigerCharge")) if atom.HasProp("_GasteigerCharge") else 0.0

        feat =[]
        feat += one_hot_atom_type(atom.GetAtomicNum())
        feat += one_hot_hybridization(atom.GetHybridization())
        feat.append(float(atom.GetIsAromatic()))
        feat += one_hot_total_h(atom.GetTotalNumHs())
        feat += one_hot_formal_charge(atom.GetFormalCharge())
        feat.append(scale_signed(g_charge, MAX_GASTEIGER_ABS))
        feat += one_hot_degree(atom.GetDegree())
        feat.append(float(atom.IsInRing()))

        neighbors = [0.0] * NUM_NEIGHBOR_FEATURES
        for n in atom.GetNeighbors():
            z = n.GetAtomicNum()
            idx = NEIGHBOR_ATOMS_IDX.get(z, None)
            if idx is not None:
                neighbors[idx] += 1.0
        neighbors =[scale_count(v, MAX_NEIGHBOR_COUNT) for v in neighbors]
        feat += neighbors

        funcs =[1.0 if i in func_group_matches[name] else 0.0 for name in FUNCTIONAL_GROUP_PATTERNS.keys()]
        feat += funcs

        atom_features.append(feat)

    x_atom = torch.tensor(atom_features, dtype=torch.float)
    if x_atom.shape[1] != TOTAL_FEATURE_DIMENSION:
        raise ValueError(f"Atom feature dim mismatch: got {x_atom.shape[1]}, expected {TOTAL_FEATURE_DIMENSION}")

    # -------------------------
    # Atom-graph edges + directed-bond nodes (line-graph nodes)
    # -------------------------
    directed_edges: List[Tuple[int, int]] = []
    edge_features: List[List[float]] = []

    atom_edge_index: List[List[int]] =[]
    atom_edge_attr: List[List[float]] =[]

    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        bond_feat =[
            float(bt == Chem.rdchem.BondType.SINGLE),
            float(bt == Chem.rdchem.BondType.DOUBLE),
            float(bt == Chem.rdchem.BondType.TRIPLE),
            float(bt == Chem.rdchem.BondType.AROMATIC),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
        ]

        atom_edge_index.extend([[u, v],[v, u]])
        atom_edge_attr.extend([bond_feat, bond_feat])

        directed_edges.extend([(u, v), (v, u)])
        edge_features.extend([bond_feat, bond_feat])

    if len(atom_edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, NUM_BOND_FEATURES), dtype=torch.float)
    else:
        edge_index = torch.tensor(atom_edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(atom_edge_attr, dtype=torch.float)

    # Line-graph nodes: x_line(u->v) = concat(x_atom[u], bond_feat(u,v))
    if len(edge_features) == 0:
        x_line = torch.empty((0, LINE_NODE_FEATURE_DIM), dtype=torch.float)
        line_src = torch.empty((0,), dtype=torch.long)
        line_dst = torch.empty((0,), dtype=torch.long)
    else:
        bond_feat_tensor = torch.tensor(edge_features, dtype=torch.float)
        line_src = torch.tensor([u for (u, _) in directed_edges], dtype=torch.long)
        line_dst = torch.tensor([v for (_, v) in directed_edges], dtype=torch.long)
        x_line = torch.cat([x_atom[line_src], bond_feat_tensor], dim=1)

        if x_line.shape[1] != LINE_NODE_FEATURE_DIM:
            raise ValueError(f"Line-node feature dim mismatch: got {x_line.shape[1]}, expected {LINE_NODE_FEATURE_DIM}")

    # -------------------------
    # Line-graph edges (directed, no-backtracking)
    # -------------------------
    if len(directed_edges) == 0:
        line_edge_index = torch.empty((2, 0), dtype=torch.long)
        line_edge_attr = torch.empty((0, NUM_LINE_EDGE_FEATURES), dtype=torch.float)
    else:
        outgoing: List[List[int]] = [[] for _ in range(num_atoms)]
        for idx, (u, _) in enumerate(directed_edges):
            outgoing[u].append(idx)

        hyb_feat_by_atom =[one_hot_hybridization(mol.GetAtomWithIdx(a).GetHybridization()) for a in range(num_atoms)]

        line_edge_indices: List[List[int]] = []
        line_edge_attrs: List[List[float]] =[]

        for i, (u_i, v_i) in enumerate(directed_edges):
            hyb_feat = hyb_feat_by_atom[v_i]
            for j in outgoing[v_i]:
                _, v_j = directed_edges[j]
                if v_j == u_i:  # backtracking
                    continue
                le_feat = hyb_feat + edge_features[i] + edge_features[j]
                line_edge_indices.append([i, j])
                line_edge_attrs.append(le_feat)

        if line_edge_indices:
            line_edge_index = torch.tensor(line_edge_indices, dtype=torch.long).t().contiguous()
            line_edge_attr = torch.tensor(line_edge_attrs, dtype=torch.float)
        else:
            line_edge_index = torch.empty((2, 0), dtype=torch.long)
            line_edge_attr = torch.empty((0, NUM_LINE_EDGE_FEATURES), dtype=torch.float)

    if line_edge_attr.numel() != 0 and line_edge_attr.shape[1] != NUM_LINE_EDGE_FEATURES:
        raise ValueError(f"Line-edge feature dim mismatch: got {line_edge_attr.shape[1]}, expected {NUM_LINE_EDGE_FEATURES}")

    # -------------------------
    # Global (graph-level) features
    # -------------------------
    try:
        g = compute_global_features(mol)  # (1, GLOBAL_FEATURE_DIM)
    except Exception as e:
        logging.debug("Global feature computation failed for SMILES=%s (%s). Using zeros.", smiles, e)
        g = torch.zeros((1, GLOBAL_FEATURE_DIM), dtype=torch.float)

    y = torch.tensor([labels], dtype=torch.float)

    mol_id_val = int(mol_id) if mol_id is not None else -1
    mol_id_t = torch.tensor([mol_id_val], dtype=torch.long)

    atom_graph = Data(x=x_atom, edge_index=edge_index, edge_attr=edge_attr, y=y, g=g, mol_id=mol_id_t)
    line_graph = Data(x=x_line, edge_index=line_edge_index, edge_attr=line_edge_attr, y=y, src=line_src, dst=line_dst, g=g, mol_id=mol_id_t)
    return atom_graph, line_graph

def load_and_preprocess_qm9_data() -> Tuple[List[Tuple[Data, Data]], int]:
    df = pd.read_csv(CSV_PATH)
    if MAX_SAMPLES is not None:
        df = df.iloc[:MAX_SAMPLES].copy()

    data_list: List[Tuple[Data, Data]] =[]
    for row_idx, row in enumerate(df.itertuples(index=False)):
        smiles = getattr(row, CSV_SMILES_COL, None)
        target = getattr(row, CSV_TARGET_COL, None)

        mol_id = getattr(row, CSV_INDEX_COL, None)
        if mol_id is None:
            mol_id = int(row_idx)

        if not isinstance(smiles, str) or not smiles:
            continue
        if target is None or (isinstance(target, float) and np.isnan(target)):
            continue

        res = smiles_to_graph_data(smiles, [float(target)], mol_id=mol_id)
        if res is not None:
            data_list.append(res)

    if not data_list:
        logging.warning("No valid molecules parsed from %s", CSV_PATH)

    return data_list, TOTAL_FEATURE_DIMENSION


def calculate_global_scaling_params(
    data_list: List[Tuple[Data, Data]],
    train_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    if not train_indices:
        raise ValueError("train_indices is empty; cannot fit global scaler.")

    xs =[]
    for idx in train_indices:
        atom_g, _ = data_list[idx]
        if not hasattr(atom_g, "g") or atom_g.g is None:
            continue
        g = atom_g.g.detach().cpu().numpy()
        g = g.reshape(-1, GLOBAL_FEATURE_DIM)  # (1, D)
        xs.append(g)

    if not xs:
        mean = np.zeros((GLOBAL_FEATURE_DIM,), dtype=np.float32)
        std = np.ones((GLOBAL_FEATURE_DIM,), dtype=np.float32)
        return mean, std

    X = np.concatenate(xs, axis=0).astype(np.float32, copy=False)  # (N_train, D)
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    std = np.where(std < 1e-12, 1.0, std).astype(np.float32, copy=False)
    mean = mean.astype(np.float32, copy=False)
    return mean, std


def apply_global_scaling(
    data_list: List[Tuple[Data, Data]],
    mean: np.ndarray,
    std: np.ndarray,
    clip_value: Optional[float] = None,
) -> None:
    mean_t = torch.tensor(mean, dtype=torch.float).view(1, -1)
    std_t = torch.tensor(std, dtype=torch.float).view(1, -1)

    do_clip = clip_value is not None and float(clip_value) > 0
    clip_min = -float(clip_value) if do_clip else None
    clip_max = float(clip_value) if do_clip else None

    for atom_g, line_g in data_list:
        if hasattr(atom_g, "g") and atom_g.g is not None:
            g = (atom_g.g - mean_t) / std_t
            atom_g.g = torch.clamp(g, clip_min, clip_max) if do_clip else g
        if hasattr(line_g, "g") and line_g.g is not None:
            g = (line_g.g - mean_t) / std_t
            line_g.g = torch.clamp(g, clip_min, clip_max) if do_clip else g

def calculate_label_scaling_params(
    data_list: List[Tuple[Data, Data]], train_indices: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    train_labels = np.concatenate([data_list[i][0].y.numpy() for i in train_indices], axis=0)  # (N, 1)
    median = np.nanmedian(train_labels, axis=0)
    q1, q3 = np.nanpercentile(train_labels, [25, 75], axis=0)
    iqr = np.where((q3 - q1) < 1e-9, 1.0, q3 - q1)
    return median, iqr

def apply_label_scaling(data_list: List[Tuple[Data, Data]], median: np.ndarray, iqr: np.ndarray) -> None:
    for atom_g, line_g in data_list:
        scaled_y = torch.tensor((atom_g.y.numpy() - median) / iqr, dtype=torch.float)
        atom_g.y = scaled_y
        line_g.y = scaled_y

def inverse_scale_labels(scaled: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    return scaled * iqr + median

def create_dataloaders(
    all_data: List[Tuple[Data, Data]],
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader([all_data[i] for i in train_idx], batch_size=batch_size, shuffle=True),
        DataLoader([all_data[i] for i in val_idx], batch_size=batch_size, shuffle=False),
        DataLoader([all_data[i] for i in test_idx], batch_size=batch_size, shuffle=False),
    )