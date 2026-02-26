"""Feature schema and encoding helpers.

This module defines:
- functional-group multi-hot features (SMARTS patterns)
- neighbor atom-type count features
- atom categorical features encoded as one-hot (with an "other" bucket)
- light scaling helpers for continuous/count features
- bond features and derived line-graph feature dimensions

Design notes:
- One-hot is applied to categorical/discrete features to avoid imposing a fake ordinal structure
  (e.g., RDKit hybridization enums, formal charge categories).
- Continuous features remain continuous (e.g., Gasteiger charge), with optional clipping/scaling.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set

from rdkit import Chem
from rdkit.Chem import Descriptors

# -------------------------
# Functional groups (multi-hot)
# -------------------------

FUNCTIONAL_GROUP_SMARTS: Dict[str, str] = {
    "Nitro": "[N+](=O)[O-]",
    "Sulfone": "[#16](=[OX1])(=[OX1])",
    "CF3": "C(F)(F)F",
    "C#N": "[CX2]#[NX1]",
    "N-C#N": "[#7,n]C#N",
    "C=O": "[CX3]=[OX1]",
    "C=N": "[CX3]=[NX2]",
    "O-C=N": "[#8][CX3]=[NX2]",
    "N-C=N": "[#7][CX3]=[NX2]",
    "Amine_Pri": "[NX3;H2]",
    "Amine_Sec": "[NX3;H1]",
    "Amine_Tert": "[NX3;H0]",
    "Ether": "[OD2]([#6])[#6]",
    "OH": "[OX2H]",
    "CH3": "[CX4H3]",
    "Pyridine_N": "[n&D2]",
    "Pyrrole_N": "[n&D3]",
}

FUNCTIONAL_GROUP_PATTERNS: Dict[str, Chem.Mol] = {}
for name, smarts in FUNCTIONAL_GROUP_SMARTS.items():
    pat = Chem.MolFromSmarts(smarts)
    if pat is None:
        logging.warning("Could not parse SMARTS for functional group '%s': %s", name, smarts)
    else:
        FUNCTIONAL_GROUP_PATTERNS[name] = pat

NUM_FUNC_GROUPS: int = len(FUNCTIONAL_GROUP_PATTERNS)

# -------------------------
# Neighbor atom-type counts
# -------------------------

NEIGHBOR_ATOMS_SET: Set[int] = {5, 6, 7, 8, 9, 15, 16, 17}  # B, C, N, O, F, P, S, Cl
NEIGHBOR_ATOMS_LIST: List[int] = sorted(NEIGHBOR_ATOMS_SET)
NUM_NEIGHBOR_FEATURES: int = len(NEIGHBOR_ATOMS_LIST)
NEIGHBOR_ATOMS_IDX: Dict[int, int] = {Z: i for i, Z in enumerate(NEIGHBOR_ATOMS_LIST)}

# -------------------------
# Categorical feature vocabularies (one-hot)
# -------------------------

# Element types to one-hot; includes common organic/halogen elements plus an "other" bucket.
ATOM_TYPES: List[int] =[1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H, B, C, N, O, F, Si, P, S, Cl, Br, I
ATOM_TYPE_OTHER_INDEX: int = len(ATOM_TYPES)
ATOM_TYPE_DIM: int = len(ATOM_TYPES) + 1

HYBRIDIZATION_TYPES: List[Chem.rdchem.HybridizationType] =[
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
HYB_OTHER_INDEX: int = len(HYBRIDIZATION_TYPES)
HYB_DIM: int = len(HYBRIDIZATION_TYPES) + 1

MAX_TOTAL_H: int = 6
TOTAL_H_DIM: int = (MAX_TOTAL_H + 1) + 1  # 0..MAX_TOTAL_H plus "other"

MAX_DEGREE: int = 6
DEGREE_DIM: int = (MAX_DEGREE + 1) + 1  # 0..MAX_DEGREE plus "other"

FORMAL_CHARGE_CATEGORIES: List[int] = [-3, -2, -1, 0, 1, 2, 3]
FORMAL_CHARGE_OTHER_INDEX: int = len(FORMAL_CHARGE_CATEGORIES)
FORMAL_CHARGE_DIM: int = len(FORMAL_CHARGE_CATEGORIES) + 1

# Continuous feature scaling (kept continuous, optionally clipped)
MAX_GASTEIGER_ABS: float = 2.0

# Neighbor count scaling
MAX_NEIGHBOR_COUNT: int = MAX_DEGREE

# -------------------------
# One-hot / scaling helpers
# -------------------------

def _one_hot_from_list(value, categories) -> List[float]:
    """One-hot encode `value` against `categories` with a final 'other' bucket."""
    out = [0.0] * (len(categories) + 1)
    idx = None
    for i, c in enumerate(categories):
        if value == c:
            idx = i
            break
    if idx is None:
        idx = len(categories)
    out[idx] = 1.0
    return out

def one_hot_atom_type(atomic_num: int) -> List[float]:
    return _one_hot_from_list(int(atomic_num), ATOM_TYPES)

def one_hot_hybridization(hyb: Chem.rdchem.HybridizationType) -> List[float]:
    return _one_hot_from_list(hyb, HYBRIDIZATION_TYPES)

def one_hot_total_h(total_h: int) -> List[float]:
    # 0..MAX_TOTAL_H else other
    cats = list(range(0, MAX_TOTAL_H + 1))
    return _one_hot_from_list(int(total_h), cats)

def one_hot_degree(degree: int) -> List[float]:
    cats = list(range(0, MAX_DEGREE + 1))
    return _one_hot_from_list(int(degree), cats)

def one_hot_formal_charge(q: int) -> List[float]:
    return _one_hot_from_list(int(q), FORMAL_CHARGE_CATEGORIES)

def scale_signed(val: float, max_abs: float) -> float:
    """Scale signed value into [-1, 1] with clipping."""
    if max_abs <= 0:
        return float(val)
    if val > max_abs:
        val = max_abs
    elif val < -max_abs:
        val = -max_abs
    return float(val / max_abs)

def scale_count(val: float, max_count: float) -> float:
    """Scale nonnegative count into [0, 1] with clipping."""
    if max_count <= 0:
        return float(val)
    if val < 0:
        val = 0.0
    if val > max_count:
        val = max_count
    return float(val / max_count)

# -------------------------
# Atom feature layout (encoded)
# -------------------------
TOTAL_FEATURE_DIMENSION: int = (
    ATOM_TYPE_DIM
    + HYB_DIM
    + 1
    + TOTAL_H_DIM
    + FORMAL_CHARGE_DIM
    + 1
    + DEGREE_DIM
    + 1
    + NUM_NEIGHBOR_FEATURES
    + NUM_FUNC_GROUPS
)

# -------------------------
# Bond / line-graph features
# -------------------------
NUM_BOND_FEATURES: int = 6
LINE_NODE_FEATURE_DIM: int = TOTAL_FEATURE_DIMENSION + NUM_BOND_FEATURES
NUM_LINE_EDGE_FEATURES: int = HYB_DIM + NUM_BOND_FEATURES + NUM_BOND_FEATURES

# -------------------------
# Global (graph-level) features (RDKit 2D descriptors)
# -------------------------
# [수정됨] RDKit의 모든 2D Descriptor 개수(약 200여 개)를 동적으로 할당
GLOBAL_FEATURE_DIM: int = len(Descriptors.descList)