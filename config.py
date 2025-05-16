# --- START OF FILE config.py ---
"""
Central configuration file for the LOHC GNN project.
Focusing on MULTI TARGET PREDICTION: Dehydrogenated_energy, Hydrogenated_energy, Potential.
Using GAT with Max Pooling and enhanced features.
Enhanced features include:
- Distance to heteroatom types (N, O, S, B)
- Counts of directly bonded heteroatoms (N, O, S, P, F, Cl, Br, I, B) per atom
- Counts of neighboring atoms by type
- Counts of specific chemical motifs per molecule
- Features indicating if an atom is part of specific functional groups
"""

import torch
from rdkit import Chem
import os
import logging
from typing import Optional

# --- Data Configuration ---
DATA_FILE_PATH = 'lohc_data.xlsx' # As per user's file name
DEHYDRO_SMILES_COL = 'Dehydrogenated_SMILES'
HYDRO_SMILES_COL = 'Hydrogenated_SMILES'
# --- MODIFIED: Target labels for multi-target prediction ---
LABEL_COLS = ['Dehydrogenated_energy', 'Hydrogenated_energy', 'Potential']
PROPERTY_NAMES = ['Dehydrogenated_energy', 'Hydrogenated_energy', 'Potential']
# ---------------------------------------------------------
# --- REMOVED: RDS_CARBON_INDEX_COL is no longer needed ---
# -----------------------------------------

# --- Model Configuration ---
# --- MODIFIED: Model save path for multi-target prediction ---
MODEL_SAVE_PATH = 'lohc_paired_gat_maxpool_multiTarget_enhancedFeats_funcAtom.pth'
LABEL_SCALING_PARAMS_PATH = 'label_scaling_params_multiTarget_enhancedFeats_funcAtom.npz'
# -----------------------------------------------------------

# --- Training Hyperparameters ---
HYPERPARAMS = {
    'batch_size': 16,
    'learning_rate': 0.001,
    'epochs': 300, # Adjust as needed
    'hidden_channels': 128,
    'dropout_rate': 0.5,
    'test_split_ratio': 0.2,
    'random_state': 42,
    # --- GAT Specific Hyperparameters ---
    'gat_heads': 4,
    'gat_output_heads': 1, # Usually 1 for final GAT layer before pooling/MLP
    'gnn_layers': 3,
    # ------------------------------------
}
# Derived from LABEL_COLS - automatically updated
HYPERPARAMS['num_output_features'] = len(LABEL_COLS) # Will be 3


# --- Feature Engineering Configuration ---

# --- Basic Atom Features & Distance Features ---
HETEROATOMS_N = {7}
HETEROATOMS_O = {8}
HETEROATOMS_S = {16}
HETEROATOMS_B = {5}
MAX_HETERO_DIST = 10.0 # Max distance for BFS search

# --- Feature A: Directly Bonded Heteroatoms ---
DIRECT_HETEROATOMS_SET = {5, 7, 8, 9, 15, 16, 17, 35, 53} # B, N, O, F, P, S, Cl, Br, I
DIRECT_HETEROATOMS_LIST = sorted(list(DIRECT_HETEROATOMS_SET))
NUM_DIRECT_HETERO_FEATURES = len(DIRECT_HETEROATOMS_LIST) # Should be 9
DIRECT_HETEROATOMS_IDX = {num: i for i, num in enumerate(DIRECT_HETEROATOMS_LIST)}


# --- Neighbor Atom Counts Feature ---
NEIGHBOR_ATOM_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B']
NEIGHBOR_ATOM_NUMS = {symbol: Chem.GetPeriodicTable().GetAtomicNumber(symbol) for symbol in NEIGHBOR_ATOM_SYMBOLS}
NEIGHBOR_ATOM_IDX = {num: i for i, num in enumerate(NEIGHBOR_ATOM_NUMS.values())}
NUM_NEIGHBOR_FEATURES = len(NEIGHBOR_ATOM_SYMBOLS) # Should be 10

# --- Molecular Descriptors (including Motifs) ---
FUNCTIONAL_GROUP_SMARTS = {
    'CH3': '[CX4H3]', 'NH2': '[NX3H2]', 'OH': '[OX2H]', 'F': '[F]',
    'Cl': '[Cl]', 'COOH': '[CX3](=[OX1])[OX2H1]', 'C=O': '[CX3]=[OX1]',
    'C-N': '[CX4][NX3]', 'C=N': '[CX3]=[NX2]', 'C#N': '[CX2]#[NX1]'
}
FUNCTIONAL_GROUP_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in FUNCTIONAL_GROUP_SMARTS.items()}
NUM_FUNC_GROUPS = len(FUNCTIONAL_GROUP_PATTERNS) # Should be 10

IMPORTANT_MOTIFS_SMARTS = {
    'NCN_sp3': '[#7]-[CX4]-[#7]',
    'OCN_sp3': '[#8]-[CX4]-[#7]',
    'Hetero_alpha_C': '[#7,#8,#16,#15]-[CX4]',
    'OCN_sp2': '[#8]-[CX3]-[#7]',
    'NCN_sp2': '[#7]-[CX3]-[#7]',
    'O_eq_C_N': '[#8X1]=[#6X3]-[#7]',
    'N_eq_C_N': '[#7X1]#[#6X2]-[#7]',
    'N_dash_N_C': '[#7]-[#7]-[#6]',
}
IMPORTANT_MOTIFS_PATTERNS = {}
for name, smarts in IMPORTANT_MOTIFS_SMARTS.items():
    mol_pattern = Chem.MolFromSmarts(smarts)
    if mol_pattern:
        IMPORTANT_MOTIFS_PATTERNS[name] = mol_pattern
    else:
        try:
            logging.warning(f"Could not parse SMARTS for motif '{name}': {smarts}")
        except NameError: # Handle case where logging might not be configured yet
             print(f"Warning: Could not parse SMARTS for motif '{name}': {smarts}")
NUM_IMPORTANT_MOTIFS = len(IMPORTANT_MOTIFS_PATTERNS) # Should be 8

# --- NEW: Functional Group Atom Features Configuration ---
NUM_FUNC_GROUP_ATOM_FEATURES = NUM_FUNC_GROUPS # Should be 10
# -------------------------------------------------------

# --- Feature Scaling Configuration ---
# The order MUST match the concatenation order in data_preprocessing.py:
# 1. Basic Atom Features (12)
# 2. Distance Features (4)
# 3. Direct Heteroatom Counts (NUM_DIRECT_HETERO_FEATURES = 9)
# 4. Functional Group Atom Features (NUM_FUNC_GROUP_ATOM_FEATURES = 10)
# 5. Neighbor Atom Counts (NUM_NEIGHBOR_FEATURES = 10)
# 6. Molecular Descriptors (basic mol(5) + func groups(10) + additional(14) + motifs(8) = 37)

ATOM_FEATURE_MAX_BASIC_DIST_DIRECT_FUNCATOM = torch.tensor([
    # Basic features (12)
    118, 6, 8, 4, 1, 2, 3, 4, 5, 1, 2.5, 1.0,
    # Distances to heteroatom types (N, O, S, B) (4)
    MAX_HETERO_DIST, MAX_HETERO_DIST, MAX_HETERO_DIST, MAX_HETERO_DIST,
    # Feature A: Direct Heteroatom Counts (9 features)
    4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
    # Functional Group Atom Features (10 features)
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
], dtype=torch.float)
NUM_ATOM_BASIC_DIST_DIRECT_FUNCATOM = ATOM_FEATURE_MAX_BASIC_DIST_DIRECT_FUNCATOM.shape[0] # 12 + 4 + 9 + 10 = 35

NEIGHBOR_FEATURE_MAX = torch.tensor([
    # Feature: Neighbor Atom Counts (10 features)
    4.0] * NUM_NEIGHBOR_FEATURES, dtype=torch.float)

MOL_DESCRIPTOR_MAX_VALUES = torch.tensor([
    # Basic Mol Descriptors (5)
    10, 5, 5, 5, 5,
    # Functional Groups (10)
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    # Additional Descriptors (14)
    10, 150, 20, 15, 15, 15, 15, 10, 10, 1, 200, 100, 100, 100,
    # Important Motifs (8)
    5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
], dtype=torch.float)
NUM_MOL_DESCRIPTORS_TOTAL = MOL_DESCRIPTOR_MAX_VALUES.shape[0] # 5 + 10 + 14 + 8 = 37

# --- Final Feature Dimension Calculation ---
TOTAL_FEATURE_DIMENSION = (NUM_ATOM_BASIC_DIST_DIRECT_FUNCATOM + # 35
                           NUM_NEIGHBOR_FEATURES +               # 10
                           NUM_MOL_DESCRIPTORS_TOTAL)            # 37
# Expected: 35 + 10 + 37 = 82

# --- Global variables for scaling parameters (initialized in data_preprocessing) ---
FEATURE_MIN_VALUES: Optional[torch.Tensor] = None
FEATURE_MAX_VALUES: Optional[torch.Tensor] = None

# --- SHAP Analysis Configuration ---
SHAP_CONFIG = { 'max_samples': 150, 'nsamples': 100, 'max_display_features': 40 }

# --- Logging Configuration ---
LOGGING_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# --- END OF FILE config.py ---