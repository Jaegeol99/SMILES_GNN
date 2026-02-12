"""
Feature configuration aligned with data_processing.py.

This module defines:
  - atom feature schema (basic + neighbor-count + functional-group flags)
  - dimensions and index maps used by data_processing.py
  - scaling constants

Notes on scaling:
  * data_processing.py currently applies a legacy scaling:
        x_scaled = clamp(x_atom / ATOM_FEATURE_MAX_VALUES, 0, 1)
    This will destroy sign information for signed features (formal charge, Gasteiger charge).
  * To mitigate this without changing the feature vector layout, this file also provides
    ATOM_FEATURE_MIN_VALUES_SAFE / ATOM_FEATURE_MAX_VALUES_SAFE and a helper
    `scale_atom_features_safe(x_atom)` that performs min-max scaling into [0, 1]
    while preserving sign.
    Activating it requires a 1-line change in data_processing.py.
"""

import logging
from typing import List, Dict, Set

import torch
from rdkit import Chem


FUNCTIONAL_GROUP_SMARTS: Dict[str, str] = {
    'Nitro': '[N+](=O)[O-]', 
    'Sulfone': '[#16](=[OX1])(=[OX1])', 
    'CF3': 'C(F)(F)F', 
    'C#N': '[CX2]#[NX1]', 
    'N-C#N': '[#7,n]C#N', 
    'C=O': '[CX3]=[OX1]', 
    'C=N': '[CX3]=[NX2]', 
    'O-C=N': '[#8][CX3]=[NX2]', 
    'N-C=N': '[#7][CX3]=[NX2]', 
    'Amine_Pri': '[NX3;H2]', 
    'Amine_Sec': '[NX3;H1]', 
    'Amine_Tert': '[NX3;H0]', 
    'Ether': '[OD2]([#6])[#6]', 
    'OH': '[OX2H]', 'CH3': '[CX4H3]', 
    'Pyridine_N': '[n&D2]', 
    'Pyrrole_N': '[n&D3]', 
}

FUNCTIONAL_GROUP_PATTERNS: Dict[str, Chem.Mol] = {}
for name, smarts in FUNCTIONAL_GROUP_SMARTS.items():
    pat = Chem.MolFromSmarts(smarts)
    if pat is None:
        logging.warning("Could not parse SMARTS for functional group '%s': %s", name, smarts)
    else:
        FUNCTIONAL_GROUP_PATTERNS[name] = pat

# number of functional-group indicator features (multi-hot)
NUM_FUNC_GROUPS: int = len(FUNCTIONAL_GROUP_PATTERNS)


# -------------------------
# Neighbor atom-type counts
# -------------------------
# Atom.GetNeighbors() counts are accumulated for these atomic numbers.
# Keep this list stable to avoid feature-order drift.
#
# Default set is "organic + common heteroatoms + halogens".
NEIGHBOR_ATOMS_SET: Set[int] = {
    5,   # B
    6,   # C
    7,   # N
    8,   # O
    9,   # F
    15,  # P
    16,  # S
    17,  # Cl
}
NEIGHBOR_ATOMS_LIST: List[int] = sorted(NEIGHBOR_ATOMS_SET)
NUM_NEIGHBOR_FEATURES: int = len(NEIGHBOR_ATOMS_LIST)
NEIGHBOR_ATOMS_IDX: Dict[int, int] = {Z: i for i, Z in enumerate(NEIGHBOR_ATOMS_LIST)}

# -------------------------
# Atom feature dimension
# -------------------------
# data_processing.py basic features (in this exact order):
#   0 atomic number
#   1 hybridization (RDKit enum as float; ordinal risk remains)
#   2 is aromatic (0/1)
#   3 total H count
#   4 formal charge (signed)
#   5 Gasteiger charge (signed)
#   6 degree
#   7 is in ring (0/1)
NUM_BASIC_ATOM_FEATURES: int = 8
TOTAL_FEATURE_DIMENSION: int = NUM_BASIC_ATOM_FEATURES + NUM_NEIGHBOR_FEATURES + NUM_FUNC_GROUPS

# -------------------------
# Scaling constants (legacy)
# -------------------------
# Legacy scaling in data_processing.py:
#   clamp(x_atom / ATOM_FEATURE_MAX_VALUES, 0, 1)
#
# IMPORTANT: negative values are mapped to 0 after clamp.
# Use SAFE scaling below to preserve sign for charges.

_MAX_ATOMIC_NUM = 100.0
_MAX_HYB_ENUM = 8.0
_MAX_TOTAL_H = 6.0
_MAX_FORMAL_CHARGE = 3.0
_MAX_GASTEIGER_ABS = 2.0
_MAX_DEGREE = 6.0
_MAX_NEIGHBOR_COUNT = _MAX_DEGREE
_MAX_FUNC_FLAG = 1.0


# -------------------------
# Scaling constants (safe)
# -------------------------
# Min-max scaling into [0, 1], preserving signed information for formal/Gasteiger charge:
#   x_scaled = clamp((x - min) / (max - min), 0, 1)
#
# Activating requires changing data_processing.py to call scale_atom_features_safe().

ATOM_FEATURE_MIN_VALUES_SAFE: torch.Tensor = torch.tensor(
    [
        0.0,   # atomic number
        0.0,   # hybridization enum
        0.0,   # is aromatic
        0.0,   # total H
        -_MAX_FORMAL_CHARGE,
        -_MAX_GASTEIGER_ABS,
        0.0,   # degree
        0.0,   # is in ring
    ]
    + [0.0] * NUM_NEIGHBOR_FEATURES
    + [0.0] * NUM_FUNC_GROUPS,
    dtype=torch.float,
)

ATOM_FEATURE_MAX_VALUES_SAFE: torch.Tensor = torch.tensor(
    [
        _MAX_ATOMIC_NUM,
        _MAX_HYB_ENUM,
        1.0,
        _MAX_TOTAL_H,
        _MAX_FORMAL_CHARGE,
        _MAX_GASTEIGER_ABS,
        _MAX_DEGREE,
        1.0,
    ]
    + [_MAX_NEIGHBOR_COUNT] * NUM_NEIGHBOR_FEATURES
    + [_MAX_FUNC_FLAG] * NUM_FUNC_GROUPS,
    dtype=torch.float,
)

def scale_atom_features_safe(x_atom: torch.Tensor) -> torch.Tensor:
    """Min-max scale atom features into [0, 1] with sign preservation."""
    denom = ATOM_FEATURE_MAX_VALUES_SAFE - ATOM_FEATURE_MIN_VALUES_SAFE
    denom = torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)
    x = (x_atom - ATOM_FEATURE_MIN_VALUES_SAFE) / denom
    return torch.clamp(x, 0.0, 1.0)


# -------------------------
# Edge feature dimensions
# -------------------------
# Bond feature layout in data_processing.py:
#   [single, double, triple, aromatic, is_conjugated, is_in_ring]

NUM_BOND_FEATURES: int = 6
BOND_FEATURE_MAX: torch.Tensor = torch.ones(NUM_BOND_FEATURES, dtype=torch.float)


# Line-edge feature layout in data_processing.py (length 15):
#   [hyb_is_sp, hyb_is_sp2, hyb_is_sp3] (3개) 
#   + bond_i (6개: S, D, T, Arom, Conj, Ring) 
#   + bond_j (6개: S, D, T, Arom, Conj, Ring)

NUM_LINE_EDGE_FEATURES: int = 15 
LINE_EDGE_FEATURE_MAX: torch.Tensor = torch.ones(NUM_LINE_EDGE_FEATURES, dtype=torch.float)