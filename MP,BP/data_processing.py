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
import logging
import traceback

try:
    from config import (
        DATA_FILE_PATH, SMILES_COL, LABEL_COLS
    )
    from feature_configs import *
except ImportError as e:
    logging.error(f"Failed to import configurations: {str(e)}")
    raise e

_feature_min_values: Optional[torch.Tensor] = None
_feature_max_values: Optional[torch.Tensor] = None
_feature_dimension_initialized: Optional[int] = None

PairedDataTuple = Tuple[Data, Data]

def initialize_feature_scaling(expected_feature_dim: int = TOTAL_FEATURE_DIMENSION):
    global _feature_min_values, _feature_max_values, _feature_dimension_initialized
    _feature_dimension_initialized = expected_feature_dim
    
    constructed_min_values = torch.cat([
        ATOM_FEATURE_MIN_BASIC_DIST_DIRECT_FUNCATOM,
        torch.zeros(NUM_NEIGHBOR_FEATURES),
        torch.zeros(NUM_MOL_DESCRIPTORS_TOTAL)
    ])
    
    constructed_max_values = torch.cat([
        ATOM_FEATURE_MAX_BASIC_DIST_DIRECT_FUNCATOM,
        NEIGHBOR_FEATURE_MAX,
        MOL_DESCRIPTOR_MAX_VALUES
    ])

    if constructed_max_values.shape[0] != expected_feature_dim or \
       constructed_min_values.shape[0] != expected_feature_dim:
        logging.error("Feature dimension mismatch in initialize_feature_scaling.")
        raise ValueError("Feature dimension mismatch.")
        
    _feature_min_values = constructed_min_values
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

def smiles_to_graph_data(smiles: str, labels: list) -> Optional[Tuple[Data, Data]]:
    """SMILES 문자열로부터 그래프 데이터를 생성하며, Gasteiger charge를 계산합니다."""
    if _feature_min_values is None or _feature_max_values is None:
        logging.warning("Feature scaling not initialized.")
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.warning(f"Invalid SMILES string: {smiles}")
        return None
    
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception as e:
        logging.warning(f"Could not compute Gasteiger charges for SMILES {smiles}: {e}")
        return None

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        logging.warning(f"Empty molecule for SMILES: {smiles}")
        return None
        
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
        "LipCleinskiHAcceptors": Lipinski.NumHAcceptors, "LipinskiHDonors": Lipinski.NumHDonors,
        "NumRings": rdMolDescriptors.CalcNumRings,
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3, "TPSA": rdMolDescriptors.CalcTPSA,
    }
    additional_descriptors = []
    for name, func in descriptor_calculators.items():
        try:
            val = float(func(mol))
        except Exception as e: 
            logging.warning(f"Failed to compute {name} for SMILES {smiles}: {str(e)}")
            val = 0.0
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
            float(atom.GetIsAromatic()), float(atom.GetFormalCharge()),
            float(atom.GetChiralTag()), float(atom.GetTotalNumHs()), float(atom.GetHybridization()),
            float(atom.IsInRing()), atom.GetMass() * 0.01
        ]
        
        charge = atom.GetDoubleProp('_GasteigerCharge')
        if np.isnan(charge) or np.isinf(charge):
            logging.warning(f"Invalid Gasteiger charge for atom {atom_idx} in SMILES {smiles}. Using 0.0.")
            charge = 0.0
        basic.append(charge)
        
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
        edge_attr = torch.empty((0, NUM_BOND_FEATURES), dtype=torch.float)
    
    line_edge_indices = []
    line_edge_attrs = []
    num_edges = edge_index.shape[1] // 2
    for i in range(0, len(edge_indices), 2):
        u1, v1 = edge_indices[i]
        for j in range(i + 2, len(edge_indices), 2):
            u2, v2 = edge_indices[j]
            if v1 == u2 or v1 == v2 or u1 == u2 or u1 == v2:
                line_edge_indices.extend([[i // 2, j // 2], [j // 2, i // 2]])
                line_edge_attrs.extend([[1.0], [1.0]])
            else:
                line_edge_indices.extend([[i // 2, j // 2], [j // 2, i // 2]])
                line_edge_attrs.extend([[0.0], [0.0]])
    
    if line_edge_indices:
        line_edge_index = torch.tensor(line_edge_indices, dtype=torch.long).t().contiguous()
        line_edge_attr = torch.tensor(line_edge_attrs, dtype=torch.float)
    else:
        line_edge_index = torch.empty((2, 0), dtype=torch.long)
        line_edge_attr = torch.empty((0, NUM_LINE_EDGE_FEATURES), dtype=torch.float)
    
    y = torch.tensor([labels], dtype=torch.float)
    atom_graph = Data(x=x_scaled, edge_index=edge_index, edge_attr=edge_attr, y=y)
    line_graph = Data(x=edge_attr[::2], edge_index=line_edge_index, edge_attr=line_edge_attr, y=y)
    
    return atom_graph, line_graph

def load_and_preprocess_data(
    data_path: str = DATA_FILE_PATH,
    smiles_col: str = SMILES_COL,
    label_cols: List[str] = LABEL_COLS
) -> Tuple[List[PairedDataTuple], int]:
    """엑셀 파일에서 SMILES와 라벨을 읽고, Gasteiger charge를 계산하여 그래프 데이터 리스트를 생성합니다."""
    try:
        df = pd.read_excel(data_path)
        logging.info(f"Successfully loaded data from {data_path}, shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Data file not found at {data_path}")
        return [], 0
    except Exception as e:
        logging.error(f"Error loading data file {data_path}: {e}")
        return [], 0

    initialize_feature_scaling(TOTAL_FEATURE_DIMENSION)
    feature_dimension = TOTAL_FEATURE_DIMENSION

    data_list = []
    
    for index, row in df.iterrows():
        smiles = row[smiles_col]
        
        if not isinstance(smiles, str) or not smiles.strip():
            logging.warning(f"Skipping row {index}: Invalid or empty SMILES string")
            continue

        try:
            labels_for_row = [float(x) for x in row[label_cols].tolist()]
            if any(pd.isna(label) for label in labels_for_row):
                logging.warning(f"Skipping row {index}: Missing or NaN labels")
                continue
        except (ValueError, TypeError) as e:
            logging.warning(f"Skipping row {index}: Invalid label format - {e}")
            continue

        try:
            graph_pair = smiles_to_graph_data(smiles, labels_for_row)
            if graph_pair is None:
                logging.warning(f"Skipping row {index}: Failed to process SMILES '{smiles}'")
                continue
            
            data_list.append(graph_pair)

        except Exception as e:
            logging.warning(f"Skipping row {index}: Error processing SMILES - {e}")
            traceback.print_exc()
            continue
    
    logging.info(f"Processed {len(data_list)} graph data samples")
    return data_list, feature_dimension

def calculate_label_scaling_params(
    data_list: List[PairedDataTuple],
    train_indices: List[int]
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not train_indices or not data_list:
        logging.warning("No training data provided for label scaling.")
        return None

    train_labels_list = []
    for i in train_indices:
        if i < len(data_list) and data_list[i][0].y is not None:
            train_labels_list.append(data_list[i][0].y.numpy())
    
    if not train_labels_list:
        logging.warning("No valid labels found for scaling.")
        return None

    train_labels_np = np.concatenate(train_labels_list, axis=0)

    if np.isnan(train_labels_np).any():
        logging.warning("NaN values found in labels, computing min/max with nanmin/nanmax.")
        label_min_values = np.nanmin(train_labels_np, axis=0)
        label_max_values = np.nanmax(train_labels_np, axis=0)
    else:
        label_min_values = train_labels_np.min(axis=0)
        label_max_values = train_labels_np.max(axis=0)

    label_range = label_max_values - label_min_values
    label_range[np.abs(label_range) < 1e-9] = 1.0

    logging.info(f"Label scaling parameters - Min: {label_min_values}, Max: {label_max_values}")
    return label_min_values, label_max_values

def apply_label_scaling(
    data_list: List[PairedDataTuple],
    label_min_values: Optional[np.ndarray],
    label_max_values: Optional[np.ndarray]
):
    if label_min_values is None or label_max_values is None:
        logging.warning("No label scaling parameters provided.")
        return
    
    label_range = label_max_values - label_min_values
    label_range[np.abs(label_range) < 1e-9] = 1.0

    for atom_graph, line_graph in data_list:
        if atom_graph.y is not None:
            y_np = atom_graph.y.numpy()
            scaled_y_np = (y_np - label_min_values) / label_range
            scaled_y_tensor = torch.tensor(scaled_y_np, dtype=torch.float)
            atom_graph.y = scaled_y_tensor
            line_graph.y = scaled_y_tensor.clone()
    logging.info("Label scaling applied to all graph data.")

def inverse_scale_labels(
    scaled_labels: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray
) -> np.ndarray:
    range_vals = max_vals - min_vals
    range_vals[np.abs(range_vals) < 1e-9] = 1.0
    original_labels = scaled_labels * range_vals + min_vals
    return original_labels

def create_dataloaders(
    all_data: List[PairedDataTuple],
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_data = [all_data[i] for i in train_indices if i < len(all_data)]
    val_data = [all_data[i] for i in val_indices if i < len(all_data)]
    test_data = [all_data[i] for i in test_indices if i < len(all_data)]
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)

    logging.info(f"Created data loaders: {len(train_loader.dataset)} train, {len(val_loader.dataset)} validation, {len(test_loader.dataset)} test samples")
    return train_loader, val_loader, test_loader