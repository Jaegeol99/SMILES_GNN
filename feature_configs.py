import torch
from rdkit import Chem
import logging 
from typing import List, Dict, Set, Any

HETEROATOMS_N: Set[int] = {7}
HETEROATOMS_O: Set[int] = {8}
HETEROATOMS_S: Set[int] = {16}
HETEROATOMS_B: Set[int] = {5}
MAX_HETERO_DIST: float = 10.0

DIRECT_HETEROATOMS_SET: Set[int] = {5, 7, 8, 9, 15, 16, 17, 35, 53}
DIRECT_HETEROATOMS_LIST: List[int] = sorted(list(DIRECT_HETEROATOMS_SET))
NUM_DIRECT_HETERO_FEATURES: int = len(DIRECT_HETEROATOMS_LIST)
DIRECT_HETEROATOMS_IDX: Dict[int, int] = {num: i for i, num in enumerate(DIRECT_HETEROATOMS_LIST)}

NEIGHBOR_ATOM_SYMBOLS: List[str] = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B']
NEIGHBOR_ATOM_NUMS: Dict[str, int] = {symbol: Chem.GetPeriodicTable().GetAtomicNumber(symbol) for symbol in NEIGHBOR_ATOM_SYMBOLS}
NEIGHBOR_ATOM_IDX: Dict[int, int] = {num: i for i, num in enumerate(NEIGHBOR_ATOM_NUMS.values())}
NUM_NEIGHBOR_FEATURES: int = len(NEIGHBOR_ATOM_SYMBOLS)

FUNCTIONAL_GROUP_SMARTS: Dict[str, str] = {
    'CH3': '[CX4H3]', 'NH2': '[NX3H2]', 'OH': '[OX2H]', 'F': '[F]',
    'Cl': '[Cl]', 'COOH': '[CX3](=[OX1])[OX2H1]', 'C=O': '[CX3]=[OX1]',
    'C-N': '[CX4][NX3]', 'C=N': '[CX3]=[NX2]', 'C#N': '[CX2]#[NX1]'
}
FUNCTIONAL_GROUP_PATTERNS: Dict[str, Chem.Mol] = {}
for k, v in FUNCTIONAL_GROUP_SMARTS.items():
    pattern = Chem.MolFromSmarts(v)
    if pattern:
        FUNCTIONAL_GROUP_PATTERNS[k] = pattern
    else:
        logging.warning(f"Could not parse SMARTS for functional group '{k}': {v}")
NUM_FUNC_GROUPS: int = len(FUNCTIONAL_GROUP_PATTERNS)

IMPORTANT_MOTIFS_SMARTS: Dict[str, str] = {
    'NCN_sp3': '[#7]-[CX4]-[#7]',
    'OCN_sp3': '[#8]-[CX4]-[#7]',
    'Hetero_alpha_C': '[#7,#8,#16,#15]-[CX4]',
    'OCN_sp2': '[#8]-[CX3]-[#7]',
    'NCN_sp2': '[#7]-[CX3]-[#7]',
    'O_eq_C_N': '[#8X1]=[#6X3]-[#7]',
    'N_eq_C_N': '[#7X1]#[#6X2]-[#7]',
    'N_dash_N_C': '[#7]-[#7]-[#6]',
}
IMPORTANT_MOTIFS_PATTERNS: Dict[str, Chem.Mol] = {}
for name, smarts in IMPORTANT_MOTIFS_SMARTS.items():
    mol_pattern = Chem.MolFromSmarts(smarts)
    if mol_pattern:
        IMPORTANT_MOTIFS_PATTERNS[name] = mol_pattern
        
NUM_IMPORTANT_MOTIFS: int = len(IMPORTANT_MOTIFS_PATTERNS)

NUM_FUNC_GROUP_ATOM_FEATURES: int = NUM_FUNC_GROUPS

ATOM_FEATURE_MAX_BASIC_DIST_DIRECT_FUNCATOM: torch.Tensor = torch.tensor([
    118, 6, 8, 4, 1, 2, 3, 4, 5, 1, 2.5, 1.0,
    MAX_HETERO_DIST, MAX_HETERO_DIST, MAX_HETERO_DIST, MAX_HETERO_DIST,
    4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
], dtype=torch.float)
NUM_ATOM_BASIC_DIST_DIRECT_FUNCATOM: int = ATOM_FEATURE_MAX_BASIC_DIST_DIRECT_FUNCATOM.shape[0] 

NEIGHBOR_FEATURE_MAX: torch.Tensor = torch.tensor(
    [4.0] * NUM_NEIGHBOR_FEATURES, dtype=torch.float
)

MOL_DESCRIPTOR_MAX_VALUES: torch.Tensor = torch.tensor([
    10, 5, 5, 5, 5, 
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 
    10, 150, 20, 15, 15, 15, 15, 10, 10, 1, 200, 100, 100, 100, 
    5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
], dtype=torch.float)
NUM_MOL_DESCRIPTORS_TOTAL: int = MOL_DESCRIPTOR_MAX_VALUES.shape[0]

TOTAL_FEATURE_DIMENSION: int = (
    NUM_ATOM_BASIC_DIST_DIRECT_FUNCATOM +
    NUM_NEIGHBOR_FEATURES +
    NUM_MOL_DESCRIPTORS_TOTAL
)