# config.py
import logging
from typing import List, Dict, Any

DATA_FILE_PATH: str = 'lohc_data_updated.xlsx'
DEHYDRO_SMILES_COL: str = 'Dehydrogenated_SMILES'
HYDRO_SMILES_COL: str = 'Hydrogenated_SMILES'
LABEL_COLS: List[str] = ['Potential']
PROPERTY_NAMES: List[str] = ['Standard oxidation potential']

MODEL_SAVE_PATH: str = 'lohc_model.pth'
LABEL_SCALING_PARAMS_PATH: str = 'lohc_scaler_robust.npz'

LOGGING_LEVEL: int = logging.INFO
LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'

HYPERPARAMS: Dict[str, Any] = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 150,
    'hidden_dim': 128,
    'num_layers': 4,
    'dropout_rate': 0.0,
    'test_split_ratio': 0.2,
    'random_state': 42,
    'num_output_features': len(LABEL_COLS)
}