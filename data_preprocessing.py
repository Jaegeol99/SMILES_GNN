# data_preprocessing.py
"""
Convert SMILES strings into PyTorch Geometric Data objects with enhanced atomic and molecular features,
and initialize global feature scaling parameters using configuration.
"""

import logging
from collections import deque
from typing import List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.rdmolops import GetSSSR
from torch_geometric.data import Data

# Import configuration constants
from config import (
    # Atom-level feature settings
    DIRECT_HETEROATOMS_LIST, DIRECT_HETEROATOMS_IDX, NUM_DIRECT_HETERO_FEATURES,
    NEIGHBOR_ATOM_SYMBOLS, NEIGHBOR_ATOM_IDX, NUM_NEIGHBOR_FEATURES,
    HETEROATOMS_N, HETEROATOMS_O, HETEROATOMS_S, HETEROATOMS_B, MAX_HETERO_DIST,
    # Molecular descriptor settings
    FUNCTIONAL_GROUP_PATTERNS, NUM_FUNC_GROUPS,
    IMPORTANT_MOTIFS_PATTERNS, NUM_IMPORTANT_MOTIFS,
    NUM_MOL_DESCRIPTORS_TOTAL,
    # Feature scaling
    ATOM_FEATURE_MAX_BASIC_DIST_DIRECT_FUNCATOM, NEIGHBOR_FEATURE_MAX,
    MOL_DESCRIPTOR_MAX_VALUES, TOTAL_FEATURE_DIMENSION,
    # Placeholders for scaling
    FEATURE_MIN_VALUES, FEATURE_MAX_VALUES,
    # Functional group per-atom feature count
    NUM_FUNC_GROUP_ATOM_FEATURES,
)

# Global variables for scaling, initialized via initialize_feature_scaling
_feature_min_values: Optional[torch.Tensor] = None
_feature_max_values: Optional[torch.Tensor] = None
_feature_dimension: Optional[int] = None


def initialize_feature_scaling(feature_dim: int):
    """
    Set up global feature scaling values (_feature_min_values, _feature_max_values) based
    on TOTAL_FEATURE_DIMENSION and configured max-value tensors.
    """
    global _feature_min_values, _feature_max_values, _feature_dimension

    if feature_dim != TOTAL_FEATURE_DIMENSION:
        msg = f"Feature dimension mismatch: {feature_dim} != {TOTAL_FEATURE_DIMENSION}"
        logging.error(msg)
        raise ValueError(msg)

    if _feature_dimension and _feature_dimension != feature_dim:
        logging.warning(f"Reinitializing scaling: {_feature_dimension} -> {feature_dim}")

    _feature_dimension = feature_dim
    _feature_min_values = torch.zeros(feature_dim, dtype=torch.float)

    try:
        concatenated = torch.cat([
            ATOM_FEATURE_MAX_BASIC_DIST_DIRECT_FUNCATOM,
            NEIGHBOR_FEATURE_MAX,
            MOL_DESCRIPTOR_MAX_VALUES,
        ])
        if concatenated.numel() != feature_dim:
            raise ValueError(
                f"Constructed max-values length {concatenated.numel()} != expected {feature_dim}"
            )
        _feature_max_values = concatenated
        logging.info(f"Feature scaling initialized for dimension {feature_dim}")
    except Exception as e:
        logging.error("Failed to build feature max-values", exc_info=True)
        raise


def get_shortest_distance_to_heteroatom(
    mol: Chem.Mol, start_idx: int, target_set: Set[int], max_dist: float = MAX_HETERO_DIST
) -> float:
    """
    BFS to find shortest bond path length from atom start_idx to any atom in target_set,
    capped at max_dist.
    """
    if mol.GetAtomWithIdx(start_idx).GetAtomicNum() in target_set:
        return 0.0

    visited = {start_idx}
    queue = deque([(start_idx, 0)])

    while queue:
        idx, dist = queue.popleft()
        if dist >= max_dist:
            break
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            ni = nbr.GetIdx()
            if ni in visited:
                continue
            if nbr.GetAtomicNum() in target_set:
                return float(dist + 1)
            visited.add(ni)
            queue.append((ni, dist + 1))
    return float(max_dist)


def smiles_to_graph_data(smiles: str, labels: List[float]) -> Optional[Data]:
    """
    Create a PyG Data object with node features x, edge_index, edge_attr, and label y.
    Feature order must match scaling initialization.
    """
    global _feature_min_values, _feature_max_values, _feature_dimension

    # Ensure scaling initialized
    if _feature_min_values is None or _feature_max_values is None or _feature_dimension is None:
        logging.error("Call initialize_feature_scaling() before preprocessing")
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.warning(f"Invalid SMILES: {smiles}")
        return None

    # Compute Gasteiger charges (fallback to zero on error)
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        for atom in mol.GetAtoms():
            atom.SetDoubleProp('_GasteigerCharge', 0.0)

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None

    # --- Atom-level features ---
    atom_feats = []
    # Precompute functional group matches
    func_matches = {
        name: mol.GetSubstructMatches(patt)
        for name, patt in FUNCTIONAL_GROUP_PATTERNS.items()
    }

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        # Basic atomic descriptor (e.g., atomic number, degree, charge)
        basic = [
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetTotalNumHs()),
            atom.GetIsAromatic() * 1.0,
            atom.GetMass(),  # example basic properties
        ]
        # Distance features to heteroatoms
        dists = [
            get_shortest_distance_to_heteroatom(mol, idx, target)
            for target in (HETEROATOMS_N, HETEROATOMS_O, HETEROATOMS_S, HETEROATOMS_B)
        ]
        # Direct heteroatom counts
        direct = [0.0] * NUM_DIRECT_HETERO_FEATURES
        for nbr in atom.GetNeighbors():
            an = nbr.GetAtomicNum()
            if an in DIRECT_HETEROATOMS_IDX:
                direct[DIRECT_HETEROATOMS_IDX[an]] += 1.0
        # Functional group per-atom flags
        func_atom = [0.0] * NUM_FUNC_GROUP_ATOM_FEATURES
        for i, name in enumerate(FUNCTIONAL_GROUP_PATTERNS):
            for match in func_matches[name]:
                if idx in match:
                    func_atom[i] = 1.0
                    break

        atom_feats.append(basic + dists + direct + func_atom)

    x_part1 = torch.tensor(atom_feats, dtype=torch.float)
    if x_part1.size(1) != _feature_dimension:  # first part length check
        expected = len(atom_feats[0])
        logging.error(f"Atom feat length {expected} != expected {_feature_dimension}")
        return None

    # --- Neighbor counts ---
    neighbor_feats = []
    for atom in mol.GetAtoms():
        counts = [0.0] * NUM_NEIGHBOR_FEATURES
        for nbr in atom.GetNeighbors():
            num = nbr.GetAtomicNum()
            if num in NEIGHBOR_ATOM_IDX:
                counts[NEIGHBOR_ATOM_IDX[num]] += 1.0
        neighbor_feats.append(counts)
    x_part2 = torch.tensor(neighbor_feats, dtype=torch.float)

    # --- Molecular descriptors ---
    # Example: logP, MR, rotatable bonds, HBD, HBA, ring counts, motifs
    desc_vals = [
        float(Crippen.MolLogP(mol)),
        float(Crippen.MolMR(mol)),
        float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        float(rdMolDescriptors.CalcNumHBD(mol)),
        float(rdMolDescriptors.CalcNumHBA(mol)),
        # ring-related
        float(rdMolDescriptors.CalcNumRings(mol)),
    ]
    # Functional group counts per molecule
    desc_vals += [
        float(len(mol.GetSubstructMatches(patt)))
        for patt in FUNCTIONAL_GROUP_PATTERNS.values()
    ]
    # Important motif counts
    desc_vals += [
        float(len(mol.GetSubstructMatches(patt)))
        for patt in IMPORTANT_MOTIFS_PATTERNS.values()
    ]
    mol_desc = torch.tensor(desc_vals, dtype=torch.float)
    # Repeat for each atom to match dimensions
    mol_desc_repeat = mol_desc.unsqueeze(0).repeat(num_atoms, 1)

    # --- Concatenate & Scale ---
    x_combined = torch.cat([x_part1, x_part2, mol_desc_repeat], dim=1)
    feature_range = _feature_max_values - _feature_min_values
    feature_range[feature_range == 0] = 1e-6
    x_scaled = (x_combined - _feature_min_values) / feature_range
    x_scaled = x_scaled.clamp(0.0, 1.0)

    # --- Edge features and connectivity ---
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # undirected edges
        edge_index += [(i, j), (j, i)]
        bt = bond.GetBondType()
        attrs = [
            float(bt == Chem.rdchem.BondType.SINGLE),
            float(bt == Chem.rdchem.BondType.DOUBLE),
            float(bt == Chem.rdchem.BondType.TRIPLE),
            float(bt == Chem.rdchem.BondType.AROMATIC),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            float(bond.GetBondTypeAsDouble()),
        ]
        edge_attr += [attrs, attrs]

    data = Data(
        x=x_scaled,
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        y=torch.tensor(labels, dtype=torch.float),
    )
    return data