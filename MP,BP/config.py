import os
import logging
from typing import List, Dict, Any

# Data-related configurations
DATA_FILE_PATH: str = 'melting point.xlsx'
DEHYDRO_SMILES_COL: str = 'SMILES'
HYDRO_SMILES_COL: str = 'SMILES'
LABEL_COLS: List[str] = ['Melting Point']
PROPERTY_NAMES: List[str] = ['Melting Point']
MODEL_SAVE_PATH: str = 'lohc_model_MP.pth'
LABEL_SCALING_PARAMS_PATH: str = 'lohc_scaler_MP.npz'

# Logging configuration
LOGGING_LEVEL: int = logging.INFO
LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'

# Hyperparameters for the model
HYPERPARAMS: Dict[str, Any] = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'hidden_dim': 128,
    'num_layers': 4,
    'dropout_rate': 0.5,
    'test_split_ratio': 0.2,
    'random_state': 42,
    'num_output_features': len(LABEL_COLS)
}

# SHAP configuration for model interpretation
SHAP_CONFIG: Dict[str, Any] = {
    'max_samples': 500,
    'nsamples': 100,
    'max_display_features': 10
}