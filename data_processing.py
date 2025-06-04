import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Crippen, AllChem, Lipinski
from rdkit.Chem.rdmolops import GetSSSR
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List, Optional, Tuple, Set
from collections import deque
import numpy as np
import os

try:
    from config import (
        DATA_FILE_PATH, DEHYDRO_SMILES_COL, HYDRO_SMILES_COL, LABEL_COLS
    )
    from feature_configs import (
        DIRECT_HETEROATOMS_LIST, DIRECT_HETEROATOMS_IDX, NUM_DIRECT_HETERO_FEATURES,
        NEIGHBOR_ATOM_SYMBOLS, NEIGHBOR_ATOM_IDX, NUM_NEIGHBOR_FEATURES,
        HETEROATOMS_N, HETEROATOMS_O, HETEROATOMS_S, HETEROATOMS_B, MAX_HETERO_DIST,
        FUNCTIONAL_GROUP_PATTERNS, FUNCTIONAL_GROUP_SMARTS,
        IMPORTANT_MOTIFS_PATTERNS,
        ATOM_FEATURE_MAX_BASIC_DIST_DIRECT_FUNCATOM,
        NEIGHBOR_FEATURE_MAX,
        MOL_DESCRIPTOR_MAX_VALUES,
        TOTAL_FEATURE_DIMENSION
    )
except ImportError as e:
    raise e

_feature_min_values: Optional[torch.Tensor] = None
_feature_max_values: Optional[torch.Tensor] = None
_feature_dimension_initialized: Optional[int] = None

PairedDataTuple = Tuple[Data, Data]

def initialize_feature_scaling(expected_feature_dim: int = TOTAL_FEATURE_DIMENSION):
    global _feature_min_values, _feature_max_values, _feature_dimension_initialized
    _feature_dimension_initialized = expected_feature_dim
    _feature_min_values = torch.zeros(expected_feature_dim, dtype=torch.float)
    
    constructed_max_values = torch.cat([
        ATOM_FEATURE_MAX_BASIC_DIST_DIRECT_FUNCATOM,
        NEIGHBOR_FEATURE_MAX,
        MOL_DESCRIPTOR_MAX_VALUES
    ])
    if constructed_max_values.shape[0] != expected_feature_dim:
        raise ValueError("Constructed max values dimension does not match expected dimension.")
    _feature_max_values = constructed_max_values

def get_shortest_distance_to_heteroatom(mol: Chem.Mol, start_atom_idx: int, target_atom_set: set, max_dist: float = MAX_HETERO_DIST) -> float:
    if mol.GetAtomWithIdx(start_atom_idx).GetAtomicNum() in target_atom_set:
        return 0.0
    queue = deque([(start_atom_idx, 0)])
    visited = {start_atom_idx}
    while queue:
        current_idx, distance = queue.popleft()
        if distance < max_dist:
            for neighbor in mol.GetAtomWithIdx(current_idx).GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in visited:
                    if neighbor.GetAtomicNum() in target_atom_set:
                        return float(distance + 1)
                    visited.add(neighbor_idx)
                    queue.append((neighbor_idx, distance + 1))
    return float(max_dist)

def smiles_to_graph_data(smiles: str, labels: list) -> Optional[Data]:
    if _feature_min_values is None or _feature_max_values is None or _feature_dimension_initialized is None:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None
    
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        for atom in mol.GetAtoms():
            atom.SetDoubleProp('_GasteigerCharge', 0.0)
    
    num_aromatic_rings = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    ring_atoms_count = {7: 0.0, 8: 0.0, 16: 0.0, 5: 0.0}
    for ring in Chem.GetSymmSSSR(mol):
        for atom_idx in ring:
            atomic_num = mol.GetAtomWithIdx(atom_idx).GetAtomicNum()
            if atomic_num in ring_atoms_count:
                ring_atoms_count[atomic_num] += 1.0

    functional_groups_count = {}
    for group, pattern in FUNCTIONAL_GROUP_PATTERNS.items():
        functional_groups_count[group] = float(len(mol.GetSubstructMatches(pattern))) if pattern else 0.0

    descriptor_calculators = {
        "MolLogP": Crippen.MolLogP, "MolMR": Crippen.MolMR,
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds,
        "NumHBD": rdMolDescriptors.CalcNumHBD, "NumHBA": rdMolDescriptors.CalcNumHBA,
        "LipinskiHAcceptors": Lipinski.NumHAcceptors, "LipinskiHDonors": Lipinski.NumHDonors,
        "NumRings": rdMolDescriptors.CalcNumRings,
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3, "TPSA": rdMolDescriptors.CalcTPSA,
    }
    additional_descriptors = []
    for name, func in descriptor_calculators.items():
        try:
            val = float(func(mol))
        except Exception: val = 0.0
        additional_descriptors.append(0.0 if pd.isna(val) else val)
    additional_descriptors.extend([
        float(mol.GetNumAtoms()), float(mol.GetNumHeavyAtoms()), float(mol.GetNumBonds())
    ])
    
    mol_descriptor_list = [
        num_aromatic_rings, ring_atoms_count[7], ring_atoms_count[8],
        ring_atoms_count[16], ring_atoms_count[5]
    ]
    for group in FUNCTIONAL_GROUP_SMARTS.keys():
        mol_descriptor_list.append(functional_groups_count.get(group, 0.0))
    mol_descriptor_list.extend(additional_descriptors)
    for motif in IMPORTANT_MOTIFS_PATTERNS.keys():
        motif_count = float(len(mol.GetSubstructMatches(IMPORTANT_MOTIFS_PATTERNS[motif]))) if IMPORTANT_MOTIFS_PATTERNS[motif] else 0.0
        mol_descriptor_list.append(motif_count)
    
    mol_descriptor_tensor = torch.tensor([mol_descriptor_list], dtype=torch.float)
    mol_descriptor_tensor_repeated = mol_descriptor_tensor.repeat(num_atoms, 1)
    
    atom_features = []
    neighbor_features = []
    func_group_matches = {
        group: mol.GetSubstructMatches(pattern) if pattern else [] 
        for group, pattern in FUNCTIONAL_GROUP_PATTERNS.items()
    }
    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        basic = [
            float(atom.GetAtomicNum()), float(atom.GetDegree()), float(atom.GetTotalValence()),
            float(atom.GetImplicitValence()), float(atom.GetIsAromatic()), float(atom.GetFormalCharge()),
            float(atom.GetChiralTag()), float(atom.GetTotalNumHs()), float(atom.GetHybridization()),
            float(atom.IsInRing()), atom.GetMass() * 0.01
        ]
        try: charge = float(atom.GetProp('_GasteigerCharge'))
        except Exception: charge = 0.0
        basic.append(0.0 if pd.isna(charge) else charge)
        
        dist_features = [
            get_shortest_distance_to_heteroatom(mol, atom_idx, HETEROATOMS_N),
            get_shortest_distance_to_heteroatom(mol, atom_idx, HETEROATOMS_O),
            get_shortest_distance_to_heteroatom(mol, atom_idx, HETEROATOMS_S),
            get_shortest_distance_to_heteroatom(mol, atom_idx, HETEROATOMS_B)
        ]
        
        direct_hetero = [0.0] * NUM_DIRECT_HETERO_FEATURES
        for neighbor_atom in atom.GetNeighbors():
            natomic = neighbor_atom.GetAtomicNum()
            if natomic in DIRECT_HETEROATOMS_IDX:
                direct_hetero[DIRECT_HETEROATOMS_IDX[natomic]] += 1.0
        
        func_group_flags = [
            1.0 if any(atom_idx in match for match in func_group_matches[group]) else 0.0
            for group in FUNCTIONAL_GROUP_SMARTS.keys()
        ]
        
        atom_feat = basic + dist_features + direct_hetero + func_group_flags
        atom_features.append(atom_feat)
        
        neighbor_count = [0.0] * NUM_NEIGHBOR_FEATURES
        for neighbor_atom in atom.GetNeighbors():
            natomic = neighbor_atom.GetAtomicNum()
            if natomic in NEIGHBOR_ATOM_IDX:
                neighbor_count[NEIGHBOR_ATOM_IDX[natomic]] += 1.0
        neighbor_features.append(neighbor_count)
    
    x_atom = torch.tensor(atom_features, dtype=torch.float)
    x_neighbor = torch.tensor(neighbor_features, dtype=torch.float)
    x_combined = torch.cat([x_atom, x_neighbor, mol_descriptor_tensor_repeated], dim=1)
    
    feature_range = _feature_max_values - _feature_min_values
    feature_range[feature_range == 0] = 1e-6
    x_scaled = torch.clamp((x_combined - _feature_min_values) / feature_range, 0.0, 1.0)
    
    edge_indices = []
    edge_attrs = []
    num_bond_features = 7 
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        bond_feat = [
            float(bond_type == Chem.rdchem.BondType.SINGLE), float(bond_type == Chem.rdchem.BondType.DOUBLE),
            float(bond_type == Chem.rdchem.BondType.TRIPLE), float(bond_type == Chem.rdchem.BondType.AROMATIC),
            float(bond.GetIsConjugated()), float(bond.IsInRing()), float(bond.GetStereo())
        ]
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([bond_feat, bond_feat])
    
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.float)
    
    y = torch.tensor([labels], dtype=torch.float)
    graph_data = Data(x=x_scaled, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    return graph_data

def load_and_preprocess_paired_data(
    data_path: str = DATA_FILE_PATH,
    smiles_de_col: str = DEHYDRO_SMILES_COL,
    smiles_hy_col: str = HYDRO_SMILES_COL,
    label_cols: List[str] = LABEL_COLS
) -> Tuple[List[PairedDataTuple], int]:
    
    df = pd.read_excel(data_path)

    initialize_feature_scaling(TOTAL_FEATURE_DIMENSION)
    feature_dimension = TOTAL_FEATURE_DIMENSION

    paired_graph_data_list: List[PairedDataTuple] = []
    
    for index, row in df.iterrows():
        smiles_de = row[smiles_de_col]
        smiles_hy = row[smiles_hy_col]
        
        if not (isinstance(smiles_de, str) and smiles_de.strip() and 
                isinstance(smiles_hy, str) and smiles_hy.strip()):
            continue

        try:
            labels_for_row = [float(x) for x in row[label_cols].tolist()]
            if any(pd.isna(label) for label in labels_for_row):
                continue
        except (ValueError, TypeError):
            continue

        try:
            graph_dehydro = smiles_to_graph_data(smiles_de, labels_for_row)
            graph_hydro = smiles_to_graph_data(smiles_hy, labels_for_row)
        except Exception:
            continue

        if graph_dehydro is not None and graph_hydro is not None:
            paired_graph_data_list.append((graph_dehydro, graph_hydro))
    
    return paired_graph_data_list, feature_dimension

def calculate_label_scaling_params(
    paired_data_list: List[PairedDataTuple],
    train_indices: List[int]
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not train_indices or not paired_data_list:
        return None

    train_labels_list = []
    for i in train_indices:
        if i < len(paired_data_list) and paired_data_list[i][0].y is not None:
            train_labels_list.append(paired_data_list[i][0].y.numpy())
    
    if not train_labels_list:
        return None

    train_labels_np = np.concatenate(train_labels_list, axis=0)

    if np.isnan(train_labels_np).any():
        label_min_values = np.nanmin(train_labels_np, axis=0)
        label_max_values = np.nanmax(train_labels_np, axis=0)
    else:
        label_min_values = train_labels_np.min(axis=0)
        label_max_values = train_labels_np.max(axis=0)

    label_range = label_max_values - label_min_values
    label_range[np.abs(label_range) < 1e-9] = 1.0

    return label_min_values, label_max_values

def apply_label_scaling(
    paired_data_list: List[PairedDataTuple],
    label_min_values: Optional[np.ndarray],
    label_max_values: Optional[np.ndarray]
):
    if label_min_values is None or label_max_values is None:
        return
    
    label_range = label_max_values - label_min_values
    label_range[np.abs(label_range) < 1e-9] = 1.0

    for graph_de, graph_hy in paired_data_list:
        if graph_de.y is not None:
            y_np = graph_de.y.numpy()
            scaled_y_np = (y_np - label_min_values) / label_range
            scaled_y_tensor = torch.tensor(scaled_y_np, dtype=torch.float)
            graph_de.y = scaled_y_tensor
            graph_hy.y = scaled_y_tensor.clone()

def inverse_scale_labels(
    scaled_labels: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray
) -> np.ndarray:
    range_vals = max_vals - min_vals
    range_vals[np.abs(range_vals) < 1e-9] = 1.0
    original_labels = scaled_labels * range_vals + min_vals
    return original_labels

def create_paired_dataloaders(
    all_scaled_data: List[PairedDataTuple],
    train_indices: List[int],
    test_indices: List[int],
    batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    train_data = [all_scaled_data[i] for i in train_indices if i < len(all_scaled_data)]
    test_data = [all_scaled_data[i] for i in test_indices if i < len(all_scaled_data)]
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader