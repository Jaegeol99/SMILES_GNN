# data_processing.py

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
import logging
import traceback

# config와 feature_configs 임포트는 그대로 둡니다.
try:
    from config import (
        DATA_FILE_PATH, SMILES_COL
    )
    from feature_configs import *
except ImportError as e:
    logging.error(f"Failed to import configurations: {str(e)}")
    raise e

# --- ▼▼▼ 데이터 처리 로직 수정 ▼▼▼ ---

# 기존 PairedDataTuple은 더 이상 필요 없습니다.

# smiles_to_graph_data 함수를 원자 전하를 처리하도록 수정
def smiles_to_charge_graph_data(smiles: str, charges: List[float]) -> Optional[Data]:
    # Feature scaling 초기화 확인 (기존 로직 유지)
    if _feature_min_values is None or _feature_max_values is None or _feature_dimension_initialized is None:
        logging.warning("Feature scaling not initialized.")
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.warning(f"Invalid SMILES string: {smiles}")
        return None
    
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        logging.warning(f"Empty molecule for SMILES: {smiles}")
        return None
        
    # 원자 수와 전하 값의 개수가 일치하는지 확인 (매우 중요)
    if num_atoms != len(charges):
        logging.warning(f"Atom count ({num_atoms}) and charge count ({len(charges)}) mismatch for SMILES: {smiles}. Skipping.")
        return None

    # --- 특징 추출 로직 (기존과 거의 동일) ---
    # (이 부분은 기존 코드의 smiles_to_graph_data 함수에서 그대로 가져옵니다)
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception as e:
        logging.warning(f"Failed to compute Gasteiger charges for SMILES {smiles}: {str(e)}")
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
    
    # --- 특징 추출 로직 종료 ---

    # 타겟(label) y를 노드 레벨로 설정
    y = torch.tensor(charges, dtype=torch.float).view(-1, 1)
    
    # Data 객체 생성 (line graph는 더 이상 필요 없음)
    graph_data = Data(x=x_scaled, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    return graph_data

# 메인 데이터 로딩 함수를 수정
def load_and_preprocess_data(
    data_path: str = DATA_FILE_PATH,
    smiles_col: str = SMILES_COL
) -> Tuple[List[Data], int]:
    try:
        df = pd.read_excel(data_path)
        logging.info(f"Successfully loaded data from {data_path}, shape: {df.shape}")
    except FileNotFoundError as e:
        logging.error(f"Data file not found at {data_path}: {str(e)}")
        return [], 0
    except Exception as e:
        logging.error(f"Error loading data file {data_path}: {str(e)}")
        traceback.print_exc()
        return [], 0

    initialize_feature_scaling(TOTAL_FEATURE_DIMENSION)
    feature_dimension = TOTAL_FEATURE_DIMENSION

    graph_data_list: List[Data] = []
    
    # 'Charge_'로 시작하는 모든 컬럼을 찾습니다.
    charge_cols = [col for col in df.columns if col.startswith('Charge_')]
    
    for index, row in df.iterrows():
        smiles = row[smiles_col]
        
        if not (isinstance(smiles, str) and smiles.strip()):
            logging.warning(f"Skipping row {index}: Invalid or empty SMILES string")
            continue

        # Charge 컬럼에서 NaN이 아닌 값들만 추출하여 리스트로 만듭니다.
        charges = row[charge_cols].dropna().astype(float).tolist()
        
        if not charges:
            logging.warning(f"Skipping row {index}: No valid charge data found")
            continue

        try:
            graph_data = smiles_to_charge_graph_data(smiles, charges)
            if graph_data:
                graph_data_list.append(graph_data)
            else:
                logging.warning(f"Skipping row {index}: Failed to process SMILES to graph.")
                
        except Exception as e:
            logging.warning(f"Skipping row {index}: Error processing SMILES - {str(e)}")
            traceback.print_exc()
            continue
    
    logging.info(f"Processed {len(graph_data_list)} graph data samples")
    return graph_data_list, feature_dimension

# 라벨 스케일링 함수 수정
def calculate_label_scaling_params(
    data_list: List[Data],
    train_indices: List[int]
) -> Optional[Tuple[float, float]]: # 스케일러가 단일 값(min, max)을 반환하도록 변경
    if not train_indices or not data_list:
        logging.warning("No training data provided for label scaling.")
        return None

    # 모든 학습 데이터의 모든 원자 전하 값을 하나의 리스트에 모읍니다.
    all_train_labels = []
    for i in train_indices:
        if i < len(data_list) and data_list[i].y is not None:
            all_train_labels.extend(data_list[i].y.view(-1).tolist())
    
    if not all_train_labels:
        logging.warning("No valid labels found for scaling.")
        return None

    all_train_labels_np = np.array(all_train_labels)

    # 단일 min/max 값 계산
    label_min = np.nanmin(all_train_labels_np)
    label_max = np.nanmax(all_train_labels_np)

    logging.info(f"Label scaling parameters - Min: {label_min}, Max: {label_max}")
    return float(label_min), float(label_max)

def apply_label_scaling(
    data_list: List[Data],
    label_min: Optional[float],
    label_max: Optional[float]
):
    if label_min is None or label_max is None:
        logging.warning("No label scaling parameters provided.")
        return
    
    label_range = label_max - label_min
    if abs(label_range) < 1e-9:
        label_range = 1.0

    for graph_data in data_list:
        if graph_data.y is not None:
            scaled_y = (graph_data.y - label_min) / label_range
            graph_data.y = scaled_y
    logging.info("Label scaling applied to all graph data.")

def inverse_scale_labels(
    scaled_labels: np.ndarray, min_val: float, max_val: float
) -> np.ndarray:
    range_val = max_val - min_val
    if abs(range_val) < 1e-9:
        range_val = 1.0
    original_labels = scaled_labels * range_val + min_val
    return original_labels

# 데이터 로더 생성 함수 수정
def create_dataloaders(
    all_scaled_data: List[Data],
    train_indices: List[int],
    test_indices: List[int],
    batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    train_data = [all_scaled_data[i] for i in train_indices if i < len(all_scaled_data)]
    test_data = [all_scaled_data[i] for i in test_indices if i < len(all_scaled_data)]
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)

    logging.info(f"Created data loaders: {len(train_loader.dataset)} train samples, {len(test_loader.dataset)} test samples")
    return train_loader, test_loader

# get_shortest_distance_to_heteroatom, initialize_feature_scaling 함수는 기존과 동일하게 유지합니다.
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
        logging.error("Feature dimension mismatch in initialize_feature_scaling.")
        raise ValueError("Feature dimension mismatch.")
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
# --- ▲▲▲ 데이터 처리 로직 수정 종료 ▲▲▲ ---