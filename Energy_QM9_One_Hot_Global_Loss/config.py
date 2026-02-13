import logging
from typing import List, Dict, Any

# -------------------------
# Dataset: QM9
# -------------------------
CSV_PATH: str = "qm9_valid_smiles_atomization_free_energy.csv"  # 첨부 파일명
CSV_SMILES_COL: str = "smiles"
CSV_TARGET_COL: str = "A_G_eV"   # 지금 생성한 csv 헤더 기준
CSV_INDEX_COL: str = "i"         # 있으면 로그/추적용
MAX_SAMPLES: int | None = 10000     # 전체(~130k) 다 쓰면 None, 빠른 실험이면 제한

# Target metadata
PROPERTY_NAMES: List[str] = ["QM9_G_free_energy_298K_eV"]
LABEL_COLS: List[str] = ["G"]       # 코드의 '출력 차원' 계산 편의용 (실질적으로 1개)

MODEL_SAVE_PATH: str = 'lohc_model.pth'
LABEL_SCALING_PARAMS_PATH: str = 'lohc_scaler.npz'
GLOBAL_SCALING_PARAMS_PATH: str = 'global_desc_scaler.npz'

# Logging configuration
LOGGING_LEVEL: int = logging.INFO
LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'

# Hyperparameters for the model
HYPERPARAMS: Dict[str, Any] = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'epochs': 50,

    # Loss
    # - 'mse': nn.MSELoss
    # - 'huber': nn.HuberLoss (or SmoothL1 fallback)
    'loss_type': 'huber',
    'huber_delta': 1.0,

    # Gradient clipping / debugging
    'grad_clip_norm': 1.0,     # set None or <=0 to disable
    'log_grad_norm': True,

    # Global descriptor z-score clipping (after train-fit scaling)
    'global_clip_value': 5.0,  # set None or <=0 to disable
    'use_scheduler': True,
    'scheduler_metric': 'mae',  # 'mae' or 'mse'
    'lr_factor': 0.5,
    'lr_patience': 5,
    'min_lr': 1e-6,
    'hidden_dim': 256,
    'num_layers': 4,
    'dropout_rate': 0.1,
    'test_split_ratio': 0.2,
    'random_state': 42,
    'num_output_features': 1
}

# SHAP configuration for model interpretation
SHAP_CONFIG: Dict[str, Any] = {
    'max_samples': 500,
    'nsamples': 500,
    'max_display_features': 10
}