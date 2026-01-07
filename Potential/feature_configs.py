import torch
from rdkit import Chem
import logging
from typing import List, Dict, Set, Any

DIRECT_HETEROATOMS_SET: Set[int] = {5, 7, 8, 16}
DIRECT_HETEROATOMS_LIST: List[int] = sorted(list(DIRECT_HETEROATOMS_SET))
NUM_DIRECT_HETERO_FEATURES: int = len(DIRECT_HETEROATOMS_LIST)
DIRECT_HETEROATOMS_IDX: Dict[int, int] = {num: i for i, num in enumerate(DIRECT_HETEROATOMS_LIST)}
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

ATOM_FEATURE_MAX_BASIC_DIST_DIRECT_FUNCATOM: torch.Tensor = torch.tensor([
    1, 4, # Aromatic, H Count (Atom in Ring 제거됨)
    4.0, 4.0, 4.0, 4.0, # Heteroatom coordinations
], dtype=torch.float)
NUM_ATOM_BASIC_DIST_DIRECT_FUNCATOM: int = ATOM_FEATURE_MAX_BASIC_DIST_DIRECT_FUNCATOM.shape[0]

MOL_DESCRIPTOR_MAX_VALUES: torch.Tensor = torch.tensor([
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, # Functional Groups
    150, 20, 15, 15, 10, 1, 200, # MR, RotB, HBD, HBA, Rings, Fsp³, TPSA
], dtype=torch.float)
NUM_MOL_DESCRIPTORS_TOTAL: int = MOL_DESCRIPTOR_MAX_VALUES.shape[0]

BOND_FEATURE_MAX: torch.Tensor = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float)
NUM_BOND_FEATURES: int = BOND_FEATURE_MAX.shape[0]

LINE_EDGE_FEATURE_MAX: torch.Tensor = torch.tensor([1.0], dtype=torch.float)
NUM_LINE_EDGE_FEATURES: int = LINE_EDGE_FEATURE_MAX.shape[0]

TOTAL_FEATURE_DIMENSION: int = (
    NUM_ATOM_BASIC_DIST_DIRECT_FUNCATOM +
    NUM_MOL_DESCRIPTORS_TOTAL
)