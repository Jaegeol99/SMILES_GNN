# config.py
"""
Central configuration for the LOHC GNN multi-target energy prediction project.
Contains data paths, hyperparameters, feature definitions, SMARTS patterns,
scaling criteria, model input dimensions, SHAP settings, and logging configuration.
"""

import os
import logging

import torch
from rdkit import Chem

# --- Data Configuration ---
DATA_FILE_PATH = 'lohc_data.xlsx'  # Path to Excel dataset
DEHYDRO_SMILES_COL = 'Dehydrogenated_SMILES'
HYDRO_SMILES_COL = 'Hydrogenated_SMILES'
# Target and property names for multi-target regression
LABEL_COLS = ['Dehydrogenated_energy', 'Hydrogenated_energy', 'Potential']
PROPERTY_NAMES = LABEL_COLS.copy()

# --- Output Paths ---
MODEL_SAVE_PATH = 'lohc_paired_gat_maxpool_multiTarget_enhancedFeats_funcAtom.pth'
LABEL_SCALING_PARAMS_PATH = 'label_scaling_params_multiTarget_enhancedFeats_funcAtom.npz'

# --- Training Hyperparameters ---
HYPERPARAMS = {
    'batch_size': 16,
    'learning_rate': 0.001,
    'epochs': 300,
    'hidden_channels': 128,
    'dropout_rate': 0.5,
    'test_split_ratio': 0.2,
    'random_state': 42,
    # GAT-specific
    'gat_heads': 4,
    'gat_output_heads': 1,
    'gnn_layers': 3,
}
# Automatically derive number of output features
HYPERPARAMS['num_output_features'] = len(LABEL_COLS)

# --- Heteroatom Distance Features ---
HETEROATOMS_N = {7}    # Nitrogen
HETEROATOMS_O = {8}    # Oxygen
HETEROATOMS_S = {16}   # Sulfur
HETEROATOMS_B = {5}    # Boron
MAX_HETERO_DIST = 10.0 # Max BFS distance for heteroatom

# --- Directly Bonded Heteroatoms (Atom-level Feature A) ---
DIRECT_HETEROATOMS_LIST = ['B', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
# Map atomic numbers to index in feature vector
DIRECT_HETEROATOMS_IDX = {
    5: 0,   # B
    7: 1,   # N
    8: 2,   # O
    9: 3,   # F
    15: 4,  # P
    16: 5,  # S
    17: 6,  # Cl
    35: 7,  # Br
    53: 8,  # I
}
NUM_DIRECT_HETERO_FEATURES = len(DIRECT_HETEROATOMS_LIST)

# --- Neighbor Atom Counts Feature ---
NEIGHBOR_ATOM_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B']
NEIGHBOR_ATOM_NUMS = {symbol: Chem.GetPeriodicTable().GetAtomicNumber(symbol)
                     for symbol in NEIGHBOR_ATOM_SYMBOLS}
NEIGHBOR_ATOM_IDX = {num: i for i, num in enumerate(NEIGHBOR_ATOM_NUMS.values())}
NUM_NEIGHBOR_FEATURES = len(NEIGHBOR_ATOM_SYMBOLS)

# --- Functional Group SMARTS Patterns ---
FUNCTIONAL_GROUP_SMARTS = {
    'CH3': '[CX4H3]',
    'NH2': '[NX3H2]',
    'OH': '[OX2H]',
    'SH': '[SX2H]',
    'COOH': 'C(=O)[OX2H1]',
    'NO2': '[NX3](=O)=O',
    'C=O': '[CX3]=O',
    'C=C': 'C=C',
    'C#N': '[CX2]#[NX1]',
}
FUNCTIONAL_GROUP_PATTERNS = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in FUNCTIONAL_GROUP_SMARTS.items()
}
NUM_FUNC_GROUPS = len(FUNCTIONAL_GROUP_PATTERNS)
NUM_FUNC_GROUP_ATOM_FEATURES = NUM_FUNC_GROUPS

# --- Important Molecular Motifs SMARTS ---
IMPORTANT_MOTIFS_SMARTS = {
    'NCN_sp3': '[#7]-[CX4]-[#7]',
    'C2N2_rings': 'c1nc[nH]c1',
    'aromatic_O': 'c1coccc1',
    'N_dash_N_C': '[#7]-[#7]-[#6]',
}
IMPORTANT_MOTIFS_PATTERNS = {}
for name, smarts in IMPORTANT_MOTIFS_SMARTS.items():
    patt = Chem.MolFromSmarts(smarts)
    if patt:
        IMPORTANT_MOTIFS_PATTERNS[name] = patt
    else:
        logging.warning(f"Invalid motif SMARTS '{name}': {smarts}")
NUM_IMPORTANT_MOTIFS = len(IMPORTANT_MOTIFS_PATTERNS)

# --- Molecular Descriptor Scaling Values ---
# Must match order in data_preprocessing concatenation
ATOM_FEATURE_MAX_BASIC_DIST_DIRECT_FUNCATOM = torch.tensor([
    # Basic atomic features (example placeholders for count: atomicNum, degree, charge, Hs, aromatic, mass)
    *([10.0] * 6),
    # Distances to heteroatoms (4)
    *([MAX_HETERO_DIST] * 4),
    # Direct heteroatom counts (NUM_DIRECT_HETERO_FEATURES)
    *([4.0] * NUM_DIRECT_HETERO_FEATURES),
    # Per-atom functional group flags (NUM_FUNC_GROUP_ATOM_FEATURES)
    *([1.0] * NUM_FUNC_GROUP_ATOM_FEATURES),
], dtype=torch.float)
NUM_ATOM_BASIC_DIST_DIRECT_FUNCATOM = ATOM_FEATURE_MAX_BASIC_DIST_DIRECT_FUNCATOM.numel()

NEIGHBOR_FEATURE_MAX = torch.tensor([
    *([4.0] * NUM_NEIGHBOR_FEATURES)
], dtype=torch.float)

MOL_DESCRIPTOR_MAX_VALUES = torch.tensor([
    # Basic molecular descriptors (5)
    *([10.0] * 5),
    # Functional group counts (NUM_FUNC_GROUPS)
    *([10.0] * NUM_FUNC_GROUPS),
    # Additional descriptors (14 placeholder values)
    10.0, 150.0, 20.0, 15.0, 15.0, 15.0, 15.0, 10.0, 10.0, 1.0, 200.0, 100.0, 100.0, 100.0,
    # Important motif counts (NUM_IMPORTANT_MOTIFS)
    *([5.0] * NUM_IMPORTANT_MOTIFS),
], dtype=torch.float)
NUM_MOL_DESCRIPTORS_TOTAL = MOL_DESCRIPTOR_MAX_VALUES.numel()

# --- Total GNN Input Dimension ---
TOTAL_FEATURE_DIMENSION = (
    NUM_ATOM_BASIC_DIST_DIRECT_FUNCATOM
    + NUM_NEIGHBOR_FEATURES
    + NUM_MOL_DESCRIPTORS_TOTAL
)

# --- Global Placeholders for Scaling (to be set at runtime) ---
FEATURE_MIN_VALUES: torch.Tensor = None
FEATURE_MAX_VALUES: torch.Tensor = None

# --- SHAP Analysis Settings ---
SHAP_CONFIG = {
    'max_samples': 150,
    'nsamples': 100,
    'max_display_features': 40,
}

# --- Logging Configuration ---
LOGGING_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
