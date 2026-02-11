# feature_configs.py
import torch
from rdkit import Chem
import logging
from typing import List, Dict, Set

# 1. 이웃 원자 (Neighbors)
NEIGHBOR_ATOMS_SET: Set[int] = {7, 8, 9, 16, 17} # N, O, F, S, Cl
NEIGHBOR_ATOMS_LIST: List[int] = sorted(list(NEIGHBOR_ATOMS_SET))
NUM_NEIGHBOR_FEATURES: int = len(NEIGHBOR_ATOMS_LIST)
NEIGHBOR_ATOMS_IDX: Dict[int, int] = {num: i for i, num in enumerate(NEIGHBOR_ATOMS_LIST)}

# 2. 핵심 작용기 (Functional Groups) - 총 23개
FUNCTIONAL_GROUP_SMARTS: Dict[str, str] = {
    'Nitro': '[N+](=O)[O-]', 'Sulfone': '[#16](=[OX1])(=[OX1])', 'CF3': 'C(F)(F)F', 'C#N': '[CX2]#[NX1]', 'N-C#N': '[#7,n]C#N', 
    'C=O': '[CX3]=[OX1]', 'Ar-C=O': 'c[CX3]=[OX1]', 'C=N': '[CX3]=[NX2]', 'Ar-C=N': 'c[CX3]=[NX2]', 'O-C=N': '[#8][CX3]=[NX2]', 'N-C=N': '[#7][CX3]=[NX2]', 
    'Amine_Pri': '[NX3;H2]', 'Amine_Sec': '[NX3;H1]', 'Amine_Tert': '[NX3;H0]', 'Ether': '[OD2]([#6])[#6]', 'OH': '[OX2H]', 'CH3': '[CX4H3]', 
    'Ar-OH': 'c[OH]', 'Ar-OR': 'c[OD2]', 'Pyridine_N': '[n&D2]', 'Pyrrole_N': '[n&D3]', 'Bridgehead_N': '[n&D3&R2]', 'Multi_N_Ring': '[n]1~[n]~*~*~*~1',
}

FUNCTIONAL_GROUP_PATTERNS: Dict[str, Chem.Mol] = {}
for k, v in FUNCTIONAL_GROUP_SMARTS.items():
    pattern = Chem.MolFromSmarts(v)
    if pattern: FUNCTIONAL_GROUP_PATTERNS[k] = pattern
NUM_FUNC_GROUPS: int = len(FUNCTIONAL_GROUP_PATTERNS)

# 3. Atom Feature Max Values (36개)
ATOM_FEATURE_MAX_VALUES: torch.Tensor = torch.tensor([
    100.0, 10.0, 1.0, 10.0, 5.0, 5.0, 10.0, 1.0, # Basic (8)
    5.0, 5.0, 5.0, 5.0, 5.0, # Neighbors (5)
    *[1.0]*23 # Func Groups (23)
], dtype=torch.float)
NUM_ATOM_FEATURES: int = ATOM_FEATURE_MAX_VALUES.shape[0]

# 4. Bond Feature Max Values (6개)
BOND_FEATURE_MAX: torch.Tensor = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float)
NUM_BOND_FEATURES: int = BOND_FEATURE_MAX.shape[0]

# --- ▼▼▼ [수정] 라인 그래프 엣지 특징 차원 (11개) ▼▼▼ ---
# [sp, sp2, sp3] (3) + [Bond1: S, D, T, A] (4) + [Bond2: S, D, T, A] (4) = 11
NUM_LINE_EDGE_FEATURES: int = 11
# --- ▲▲▲ [수정 완료] ▲▲▲ ---

TOTAL_FEATURE_DIMENSION: int = NUM_ATOM_FEATURES