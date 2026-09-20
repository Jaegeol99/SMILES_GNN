import os
import logging
from typing import List, Dict, Any

DATA_FILE_PATH: str = 'boiling point.xlsx' 
SMILES_COL: str = 'SMILES'
LABEL_COLS: List[str] = ['Boiling Point']

PROPERTY_NAMES: List[str] = ['Boiling Point']
MODEL_SAVE_PATH: str = 'lohc_model_BP.pth'
LABEL_SCALING_PARAMS_PATH: str = 'lohc_scaler_BP.npz'

LOGGING_LEVEL: int = logging.INFO
LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'

HYPERPARAMS: Dict[str, Any] = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 150,
    'hidden_dim': 128,
    'num_layers': 4,
    'dropout_rate': 0.5,
    'test_split_ratio': 0.2,
    'random_state': 42,
    'num_output_features': len(LABEL_COLS)
}

SHAP_CONFIG: Dict[str, Any] = {
    'max_samples': 500,
    'nsamples': 100,
    'max_display_features': 10
}