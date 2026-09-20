import os
import logging
from typing import List, Dict, Any

DATA_FILE_PATH: str = 'lohc_data.xlsx'
DEHYDRO_SMILES_COL: str = 'Dehydrogenated_SMILES'
HYDRO_SMILES_COL: str = 'Hydrogenated_SMILES'
LABEL_COLS: List[str] = ['Dehydrogenated_energy', 'Hydrogenated_energy', 'Potential','Capacity']
PROPERTY_NAMES: List[str] = ['Dehydrogenated energy', 'Hydrogenated energy', 'Standard oxidation potential','Capacity']
MODEL_SAVE_PATH: str = 'lohc_model.pth'
LABEL_SCALING_PARAMS_PATH: str = 'lohc_scaler.npz'

LOGGING_LEVEL: int = logging.INFO
LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'

HYPERPARAMS: Dict[str, Any] = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 300,
    'hidden_dim': 128,
    'num_layers': 4,
    'dropout_rate': 0.5,
    'test_split_ratio': 0.2,
    'random_state': 42,
    'num_output_features': len(LABEL_COLS)
}

SHAP_CONFIG: Dict[str, Any] = {
    'max_samples': 500,
    'nsamples': 200,
    'max_display_features': 10
}