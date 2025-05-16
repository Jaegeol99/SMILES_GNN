# --- START OF FILE data_preprocessing.py ---

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Crippen, AllChem, Lipinski
from rdkit.Chem.rdmolops import GetSSSR
import torch
from torch_geometric.data import Data
import logging
from typing import List, Optional, Tuple, Set, Dict
from collections import deque
import numpy as np

# Import configuration constants
try:
    from config import (
        # Feature A related
        DIRECT_HETEROATOMS_LIST, DIRECT_HETEROATOMS_IDX, NUM_DIRECT_HETERO_FEATURES,
        # Neighbor features related
        NEIGHBOR_ATOM_SYMBOLS, NEIGHBOR_ATOM_IDX, NUM_NEIGHBOR_FEATURES,
        # Distance features related
        HETEROATOMS_N, HETEROATOMS_O, HETEROATOMS_S, HETEROATOMS_B, MAX_HETERO_DIST,
        # Molecular descriptors related (including motifs)
        FUNCTIONAL_GROUP_PATTERNS, FUNCTIONAL_GROUP_SMARTS, NUM_FUNC_GROUPS,
        IMPORTANT_MOTIFS_PATTERNS, IMPORTANT_MOTIFS_SMARTS, NUM_IMPORTANT_MOTIFS,
        NUM_MOL_DESCRIPTORS_TOTAL,
        # NEW: Functional Group Atom Features related
        NUM_FUNC_GROUP_ATOM_FEATURES,
        # Scaling related
        ATOM_FEATURE_MAX_BASIC_DIST_DIRECT_FUNCATOM,
        NUM_ATOM_BASIC_DIST_DIRECT_FUNCATOM,
        NEIGHBOR_FEATURE_MAX,
        MOL_DESCRIPTOR_MAX_VALUES,
        TOTAL_FEATURE_DIMENSION,
        # Global scaling vars (to be set)
        FEATURE_MIN_VALUES, FEATURE_MAX_VALUES # These are module-level in config, not directly set here
    )
except ImportError as e:
    print(f"CRITICAL Error importing config constants in data_preprocessing.py: {e}")
    print("Ensure config.py is accessible and defines ALL required constants.")
    raise e

# --- Global feature scaling parameters (module level) ---
_feature_min_values: Optional[torch.Tensor] = None
_feature_max_values: Optional[torch.Tensor] = None
_feature_dimension: Optional[int] = None
# -----------------------------------------

def initialize_feature_scaling(feature_dim: int):
    """
    Initializes global feature scaling parameters using constants from config.
    Checks if the provided dimension matches the expected TOTAL_FEATURE_DIMENSION.
    Constructs the _feature_max_values tensor by concatenating max value tensors
    in the correct order.
    """
    global _feature_min_values, _feature_max_values, _feature_dimension

    if feature_dim != TOTAL_FEATURE_DIMENSION:
        logging.error(f"FATAL: Provided feature dimension ({feature_dim}) != expected config dimension ({TOTAL_FEATURE_DIMENSION}).")
        raise ValueError("Feature dimension mismatch during scaling initialization.")

    if _feature_dimension is not None:
        if _feature_dimension != feature_dim:
            logging.warning(f"Re-initializing feature scaling with different dimension: {_feature_dimension} -> {feature_dim}")

    _feature_dimension = feature_dim
    _feature_min_values = torch.zeros(feature_dim, dtype=torch.float)

    try:
        # Order: Basic/Dist/Direct/FuncAtom -> Neighbors -> Mol Descriptors
        # ATOM_FEATURE_MAX_BASIC_DIST_DIRECT_FUNCATOM already combines Basic, Dist, Direct, FuncAtom parts
        _feature_max_values = torch.cat([
            ATOM_FEATURE_MAX_BASIC_DIST_DIRECT_FUNCATOM, # Shape [NUM_ATOM_BASIC_DIST_DIRECT_FUNCATOM]
            NEIGHBOR_FEATURE_MAX,                        # Shape [NUM_NEIGHBOR_FEATURES]
            MOL_DESCRIPTOR_MAX_VALUES                    # Shape [NUM_MOL_DESCRIPTORS_TOTAL]
        ])
        if _feature_max_values.shape[0] != feature_dim:
             raise ValueError(f"Constructed _feature_max_values dim ({_feature_max_values.shape[0]}) != expected ({feature_dim}). Check config concatenation order and individual tensor shapes.")
        logging.info(f"Feature scaling parameters initialized for dimension: {feature_dim}")
    except Exception as e:
        logging.error(f"Error constructing _feature_max_values: {e}")
        logging.error("Check config.py for ATOM_FEATURE_MAX_BASIC_DIST_DIRECT_FUNCATOM, NEIGHBOR_FEATURE_MAX, MOL_DESCRIPTOR_MAX_VALUES definitions and concatenation order.")
        raise ValueError("Failed to construct feature max values tensor.")


def get_shortest_distance_to_heteroatom(mol: Chem.Mol, start_atom_idx: int, target_atom_set: Set[int], max_dist: float = MAX_HETERO_DIST) -> float:
    """Calculates shortest graph distance from start_atom to nearest atom in target_atom_set using BFS."""
    start_atom = mol.GetAtomWithIdx(start_atom_idx)
    if start_atom.GetAtomicNum() in target_atom_set:
        return 0.0
    queue = deque([(start_atom_idx, 0)])
    visited = {start_atom_idx}
    while queue:
        current_idx, distance = queue.popleft()
        if distance >= max_dist: continue
        atom = mol.GetAtomWithIdx(current_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in visited:
                if neighbor.GetAtomicNum() in target_atom_set:
                    return float(distance + 1)
                visited.add(neighbor_idx)
                if distance + 1 < max_dist:
                    queue.append((neighbor_idx, distance + 1))
    return float(max_dist)


def smiles_to_graph_data(smiles: str, labels: List[float]) -> Optional[Data]:
    """
    Converts SMILES to PyG Data object with enhanced features and applies scaling.
    Includes new features indicating if an atom is part of specific functional groups.

    Feature Order (Matches config & initialize_feature_scaling):
    1. Basic Atom Features (12)
    2. Distance Features (4)
    3. Direct Heteroatom Counts (NUM_DIRECT_HETERO_FEATURES = 9)
    4. Functional Group Atom Features (NUM_FUNC_GROUP_ATOM_FEATURES = 10)
       -> These 4 are combined into x_part_atom_level_non_neighbor
    5. Neighbor Atom Counts (NUM_NEIGHBOR_FEATURES = 10) -> x_part_neighbors
    6. Molecular Descriptors (NUM_MOL_DESCRIPTORS_TOTAL = 37) -> mol_descriptor_tensor_repeated

    Concatenation order for final x_combined:
    x_part_atom_level_non_neighbor + x_part_neighbors + mol_descriptor_tensor_repeated
    This matches the construction of ATOM_FEATURE_MAX_BASIC_DIST_DIRECT_FUNCATOM,
    NEIGHBOR_FEATURE_MAX, and MOL_DESCRIPTOR_MAX_VALUES in config and initialize_feature_scaling.
    """
    global _feature_min_values, _feature_max_values, _feature_dimension

    if _feature_min_values is None or _feature_max_values is None or _feature_dimension is None:
        logging.error("Feature scaling not initialized. Call initialize_feature_scaling() first.")
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        for atom in mol.GetAtoms(): atom.SetDoubleProp('_GasteigerCharge', 0.0)

    num_atoms_rdkit = mol.GetNumAtoms()
    if num_atoms_rdkit == 0:
        return None
    atom_indices = list(range(num_atoms_rdkit))

    # --- Part 6: Molecular Descriptors ---
    num_aromatic_rings = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    ring_atoms_count = {7: 0.0, 8: 0.0, 16: 0.0, 5: 0.0}
    try:
        sssr_rings = GetSSSR(mol)
        for ring in sssr_rings:
            for atom_index in ring:
                atom = mol.GetAtomWithIdx(atom_index)
                atomic_num = atom.GetAtomicNum()
                if atomic_num in ring_atoms_count: ring_atoms_count[atomic_num] += 1.0
    except Exception: pass

    functional_groups_count = {}
    for k, v in FUNCTIONAL_GROUP_PATTERNS.items():
        try: functional_groups_count[k] = float(len(mol.GetSubstructMatches(v)))
        except Exception: functional_groups_count[k] = 0.0

    descriptor_calculators = {
        "MolLogP": Crippen.MolLogP, "MolMR": Crippen.MolMR,
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds,
        "NumHBD": rdMolDescriptors.CalcNumHBD, "NumHBA": rdMolDescriptors.CalcNumHBA,
        "LipinskiHAcceptors": Lipinski.NumHAcceptors, "LipinskiHDonors": Lipinski.NumHDonors,
        "NumRings": rdMolDescriptors.CalcNumRings, "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3, "TPSA": rdMolDescriptors.CalcTPSA,
    }
    additional_descriptors = []
    for name, func in descriptor_calculators.items():
        try:
            value = float(func(mol))
            additional_descriptors.append(0.0 if pd.isna(value) else value)
        except Exception: additional_descriptors.append(0.0)
    additional_descriptors.extend([
        float(mol.GetNumAtoms()), float(mol.GetNumHeavyAtoms()), float(mol.GetNumBonds()),
    ])

    motif_counts = {}
    for name, pattern in IMPORTANT_MOTIFS_PATTERNS.items():
        try:
            if pattern: motif_counts[name] = float(len(mol.GetSubstructMatches(pattern)))
            else: motif_counts[name] = 0.0
        except Exception: motif_counts[name] = 0.0

    mol_descriptor_list = [
        num_aromatic_rings,
        ring_atoms_count[7], ring_atoms_count[8], ring_atoms_count[16], ring_atoms_count[5],
    ]
    for group_name in FUNCTIONAL_GROUP_SMARTS.keys():
        mol_descriptor_list.append(functional_groups_count.get(group_name, 0.0))
    mol_descriptor_list.extend(additional_descriptors)
    for motif_name in IMPORTANT_MOTIFS_PATTERNS.keys():
         mol_descriptor_list.append(motif_counts.get(motif_name, 0.0))

    mol_descriptor_tensor = torch.tensor([mol_descriptor_list], dtype=torch.float)
    if mol_descriptor_tensor.shape[1] != NUM_MOL_DESCRIPTORS_TOTAL:
        logging.error(f"Molecular descriptor dimension mismatch for {smiles}. Expected {NUM_MOL_DESCRIPTORS_TOTAL}, got {mol_descriptor_tensor.shape[1]}. Skipping.")
        return None
    mol_descriptor_tensor_repeated = mol_descriptor_tensor.repeat(num_atoms_rdkit, 1)

    # --- Atom-Level Features (Parts 1, 2, 3, 4 combined, and Part 5 separate) ---
    atom_features_part1234_list = [] # For Basic, Distance, Direct Hetero, Func Group Atom
    atom_features_part5_list = []    # For Neighbor Counts

    func_group_matches = {}
    for name, pattern in FUNCTIONAL_GROUP_PATTERNS.items():
        if pattern:
            try:
                func_group_matches[name] = mol.GetSubstructMatches(pattern)
            except Exception:
                 func_group_matches[name] = ()

    for atom_idx in atom_indices:
        atom = mol.GetAtomWithIdx(atom_idx)

        # Part 1: Basic Atom Features (12)
        basic_features = [
            float(atom.GetAtomicNum()), float(atom.GetDegree()),
            float(atom.GetTotalValence()), float(atom.GetImplicitValence()),
            float(atom.GetIsAromatic()), float(atom.GetFormalCharge()),
            float(atom.GetChiralTag()), float(atom.GetTotalNumHs()),
            float(atom.GetHybridization()), float(atom.IsInRing()),
            atom.GetMass() * 0.01
        ]
        try:
            gasteiger_charge = float(atom.GetProp('_GasteigerCharge'))
            basic_features.append(0.0 if pd.isna(gasteiger_charge) else gasteiger_charge)
        except (KeyError, ValueError): basic_features.append(0.0)

        # Part 2: Distance Features (4)
        dist_features = [
            get_shortest_distance_to_heteroatom(mol, atom_idx, HETEROATOMS_N),
            get_shortest_distance_to_heteroatom(mol, atom_idx, HETEROATOMS_O),
            get_shortest_distance_to_heteroatom(mol, atom_idx, HETEROATOMS_S),
            get_shortest_distance_to_heteroatom(mol, atom_idx, HETEROATOMS_B)
        ]

        # Part 3: Direct Heteroatom Counts (9)
        direct_hetero_counts = [0.0] * NUM_DIRECT_HETERO_FEATURES
        for neighbor in atom.GetNeighbors():
            neighbor_atomic_num = neighbor.GetAtomicNum()
            if neighbor_atomic_num in DIRECT_HETEROATOMS_IDX:
                idx = DIRECT_HETEROATOMS_IDX[neighbor_atomic_num]
                direct_hetero_counts[idx] += 1.0
        
        # Part 4: Functional Group Atom Features (10)
        func_group_atom_features = [0.0] * NUM_FUNC_GROUP_ATOM_FEATURES
        for i, group_name in enumerate(FUNCTIONAL_GROUP_SMARTS.keys()):
             if group_name in func_group_matches:
                 for match in func_group_matches[group_name]:
                     if atom_idx in match:
                         func_group_atom_features[i] = 1.0
                         break
        
        # Combine parts 1, 2, 3, 4 for this atom
        atom_features_part1234_list.append(basic_features + dist_features + direct_hetero_counts + func_group_atom_features)

        # Part 5: Neighbor Atom Counts (10)
        neighbor_counts = [0.0] * NUM_NEIGHBOR_FEATURES
        for neighbor in atom.GetNeighbors():
            neighbor_atomic_num = neighbor.GetAtomicNum()
            if neighbor_atomic_num in NEIGHBOR_ATOM_IDX:
                neighbor_counts[NEIGHBOR_ATOM_IDX[neighbor_atomic_num]] += 1.0
        atom_features_part5_list.append(neighbor_counts)

    x_part_atom_level_non_neighbor = torch.tensor(atom_features_part1234_list, dtype=torch.float) # Basic, Dist, Direct, FuncAtom
    x_part_neighbors = torch.tensor(atom_features_part5_list, dtype=torch.float) # Neighbors

    # Check dimensions of combined non-neighbor atom features
    expected_dim_part1234 = 12 + 4 + NUM_DIRECT_HETERO_FEATURES + NUM_FUNC_GROUP_ATOM_FEATURES
    if x_part_atom_level_non_neighbor.shape[1] != expected_dim_part1234:
         logging.error(f"Atom feature (basic/dist/direct/func_atom) dim mismatch for {smiles}. Expected {expected_dim_part1234}, got {x_part_atom_level_non_neighbor.shape[1]}. Skipping.")
         return None
    # Check this matches NUM_ATOM_BASIC_DIST_DIRECT_FUNCATOM from config
    if x_part_atom_level_non_neighbor.shape[1] != NUM_ATOM_BASIC_DIST_DIRECT_FUNCATOM:
        logging.error(f"Atom feature (basic/dist/direct/func_atom) dim {x_part_atom_level_non_neighbor.shape[1]} does not match config.NUM_ATOM_BASIC_DIST_DIRECT_FUNCATOM {NUM_ATOM_BASIC_DIST_DIRECT_FUNCATOM}. Skipping.")
        return None

    if x_part_neighbors.shape[1] != NUM_NEIGHBOR_FEATURES:
         logging.error(f"Atom feature (neighbors) dim mismatch for {smiles}. Expected {NUM_NEIGHBOR_FEATURES}, got {x_part_neighbors.shape[1]}. Skipping.")
         return None

    # --- Combine All Features (Parts 1-4, Part 5, Part 6) ---
    # Concatenation order must match _feature_max_values construction in initialize_feature_scaling
    x_combined = torch.cat([
        x_part_atom_level_non_neighbor, # Corresponds to ATOM_FEATURE_MAX_BASIC_DIST_DIRECT_FUNCATOM
        x_part_neighbors,               # Corresponds to NEIGHBOR_FEATURE_MAX
        mol_descriptor_tensor_repeated  # Corresponds to MOL_DESCRIPTOR_MAX_VALUES
    ], dim=1)

    if x_combined.size(1) != _feature_dimension:
        logging.error(f"Final feature dimension mismatch for {smiles}. Expected {_feature_dimension}, got {x_combined.size(1)}. Skipping.")
        return None

    feature_range = _feature_max_values - _feature_min_values
    feature_range[feature_range == 0] = 1e-6
    x_scaled = (x_combined - _feature_min_values) / feature_range
    x_scaled = torch.clamp(x_scaled, 0.0, 1.0)

    edge_index_list = []
    edge_attr_list = []
    num_bond_features = 7
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index_list.extend([(i, j), (j, i)])
        bond_type = bond.GetBondType()
        bond_features = [
            float(bond_type == Chem.rdchem.BondType.SINGLE),
            float(bond_type == Chem.rdchem.BondType.DOUBLE),
            float(bond_type == Chem.rdchem.BondType.TRIPLE),
            float(bond_type == Chem.rdchem.BondType.AROMATIC),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            float(bond.GetStereo())
        ]
        edge_attr_list.extend([bond_features, bond_features])

    if not edge_index_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    y = torch.tensor([labels], dtype=torch.float) # labels is now a list of 3 floats

    graph_data = Data(x=x_scaled, edge_index=edge_index, edge_attr=edge_attr, y=y)

    if graph_data.num_nodes != num_atoms_rdkit:
         pass

    return graph_data

# --- END OF FILE data_preprocessing.py ---