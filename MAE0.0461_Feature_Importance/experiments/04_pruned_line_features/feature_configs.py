from __future__ import annotations

from typing import Dict, List, Set

from rdkit import Chem
from rdkit.Chem import Descriptors

FUNCTIONAL_GROUP_SMARTS: Dict[str, str] = {
    "Nitro": "[N+](=O)[O-]", "Sulfone": "[#16](=[OX1])(=[OX1])", "CF3": "C(F)(F)F",
    "C#N": "[CX2]#[NX1]", "N-C#N": "[#7,n]C#N", "C=O": "[CX3]=[OX1]",
    "C=N": "[CX3]=[NX2]", "O-C=N": "[#8][CX3]=[NX2]", "N-C=N": "[#7][CX3]=[NX2]",
    "Amine_Pri": "[NX3;H2]", "Amine_Sec": "[NX3;H1]", "Amine_Tert": "[NX3;H0]",
    "Ether": "[OD2]([#6])[#6]", "OH": "[OX2H]", "CH3": "[CX4H3]",
    "Pyridine_N": "[n&D2]", "Pyrrole_N": "[n&D3]",
}

FUNCTIONAL_GROUP_PATTERNS: Dict[str, Chem.Mol] = {}
for name, smarts in FUNCTIONAL_GROUP_SMARTS.items():
    pat = Chem.MolFromSmarts(smarts)
    if pat is not None:
        FUNCTIONAL_GROUP_PATTERNS[name] = pat

NUM_FUNC_GROUPS: int = len(FUNCTIONAL_GROUP_PATTERNS)

NEIGHBOR_ATOMS_SET: Set[int] = {5, 6, 7, 8, 9, 15, 16, 17}
NEIGHBOR_ATOMS_LIST: List[int] = sorted(NEIGHBOR_ATOMS_SET)
NUM_NEIGHBOR_FEATURES: int = len(NEIGHBOR_ATOMS_LIST)
NEIGHBOR_ATOMS_IDX: Dict[int, int] = {Z: i for i, Z in enumerate(NEIGHBOR_ATOMS_LIST)}

ATOM_TYPES: List[int] =[1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
ATOM_TYPE_DIM: int = len(ATOM_TYPES) + 1

HYBRIDIZATION_TYPES: List[Chem.rdchem.HybridizationType] =[
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
HYB_DIM: int = len(HYBRIDIZATION_TYPES) + 1

MAX_TOTAL_H: int = 6
TOTAL_H_DIM: int = (MAX_TOTAL_H + 1) + 1

MAX_DEGREE: int = 6
DEGREE_DIM: int = (MAX_DEGREE + 1) + 1

FORMAL_CHARGE_CATEGORIES: List[int] = [-3, -2, -1, 0, 1, 2, 3]
FORMAL_CHARGE_DIM: int = len(FORMAL_CHARGE_CATEGORIES) + 1

MAX_GASTEIGER_ABS: float = 2.0
MAX_NEIGHBOR_COUNT: int = MAX_DEGREE

def _one_hot_from_list(value, categories) -> List[float]:
    out =[0.0] * (len(categories) + 1)
    idx = categories.index(value) if value in categories else len(categories)
    out[idx] = 1.0
    return out

def one_hot_atom_type(atomic_num: int) -> List[float]: return _one_hot_from_list(int(atomic_num), ATOM_TYPES)
def one_hot_hybridization(hyb: Chem.rdchem.HybridizationType) -> List[float]: return _one_hot_from_list(hyb, HYBRIDIZATION_TYPES)
def one_hot_total_h(total_h: int) -> List[float]: return _one_hot_from_list(int(total_h), list(range(0, MAX_TOTAL_H + 1)))
def one_hot_degree(degree: int) -> List[float]: return _one_hot_from_list(int(degree), list(range(0, MAX_DEGREE + 1)))
def one_hot_formal_charge(q: int) -> List[float]: return _one_hot_from_list(int(q), FORMAL_CHARGE_CATEGORIES)

def scale_signed(val: float, max_abs: float) -> float:
    return float(max(-max_abs, min(val, max_abs)) / max_abs) if max_abs > 0 else float(val)

def scale_count(val: float, max_count: float) -> float:
    return float(max(0.0, min(val, max_count)) / max_count) if max_count > 0 else float(val)

TOTAL_FEATURE_DIMENSION: int = (
    ATOM_TYPE_DIM + HYB_DIM + 1 + TOTAL_H_DIM + FORMAL_CHARGE_DIM + 1 + DEGREE_DIM + 1 + NUM_NEIGHBOR_FEATURES + NUM_FUNC_GROUPS
)

NUM_BOND_FEATURES: int = 6

LAPLACIAN_PE_K: int = 0
LINE_NODE_FEATURE_DIM: int = TOTAL_FEATURE_DIMENSION + NUM_BOND_FEATURES

PSEUDO_ANGLE_DIM: int = 0
RING_STRAIN_DIM: int = 6
CONJUGATION_FLOW_DIM: int = 0

NUM_LINE_EDGE_FEATURES: int = (
    HYB_DIM + NUM_BOND_FEATURES + NUM_BOND_FEATURES + RING_STRAIN_DIM
)

GLOBAL_FEATURE_DIM: int = len(Descriptors.descList)
