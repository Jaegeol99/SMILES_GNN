import os
import logging
from typing import List, Dict, Any

# Data-related configurations
DATA_FILE_PATH: str = 'lohc_data.xlsx'
DEHYDRO_SMILES_COL: str = 'Dehydrogenated_SMILES'
HYDRO_SMILES_COL: str = 'Hydrogenated_SMILES'
LABEL_COLS: List[str] = ['Dehydrogenated_energy', 'Hydrogenated_energy', 'Potential']
PROPERTY_NAMES: List[str] = ['Dehydrogenated_energy', 'Hydrogenated_energy', 'Potential']
MODEL_SAVE_PATH: str = 'lohc_model.pth'
LABEL_SCALING_PARAMS_PATH: str = 'lohc_scaler.npz'

# Logging configuration
LOGGING_LEVEL: int = logging.INFO
LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'

# Hyperparameters for the model
HYPERPARAMS: Dict[str, Any] = {
    'batch_size': 2,
    'learning_rate': 0.001,
    'epochs': 300,
    'hidden_dim': 128,
    'num_layers': 4,
    'dropout_rate': 0.5,
    'test_split_ratio': 0.2,
    'random_state': 42,
    'num_output_features': len(LABEL_COLS)
}

# SHAP configuration for model interpretation
SHAP_CONFIG: Dict[str, Any] = {
    'max_samples': 150,
    'nsamples': 100,
    'max_display_features': 40
}