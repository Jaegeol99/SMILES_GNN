import logging
from typing import Any, Dict, List, Optional

CSV_PATH: str = "qm9.csv"
CSV_SMILES_COL: str = "smiles"
CSV_TARGET_COL: str = "A_G_eV"
CSV_INDEX_COL: str = "i"

MAX_SAMPLES: Optional[int] = 20000

PROPERTY_NAMES: List[str] = ["QM9_G_free_energy_298K_eV"]
LABEL_COLS: List[str] = ["G"]

OUTPUT_DIR: str = "outputs"
MODEL_SAVE_PATH: str = f"{OUTPUT_DIR}/lohc_model.pth"
STAGE2_MODEL_SAVE_PATH: str = f"{OUTPUT_DIR}/lohc_model_stage2.pth"
LABEL_SCALING_PARAMS_PATH: str = f"{OUTPUT_DIR}/label_scaler.npz"
GLOBAL_SCALING_PARAMS_PATH: str = f"{OUTPUT_DIR}/global_desc_standard_scaler.npz"
REPORT_DIR: str = f"{OUTPUT_DIR}/reports"
FEATURE_IMPORTANCE_CSV_PATH: str = f"{REPORT_DIR}/feature_importance_block_permutation.csv"

LOGGING_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"

SEED: int = 42

HYPERPARAMS: Dict[str, Any] = {
    "batch_size": 50,
    "epochs": 20,
    "test_split_ratio": 0.2,
    "random_state": 42,

    "hidden_dim": 300,
    "num_layers": 6,
    "dropout_rate": 0.0,
    "num_output_features": 1,

    "learning_rate": 1e-3,
    "optimizer": "adamw",
    "weight_decay": 1e-4,

    "loss_type": "huber",
    "huber_delta": 1.0,

    "grad_clip_norm": 1.0,

    "scheduler": "exponential",
    "scheduler_gamma": 0.95,
    "scheduler_metric": "mae",
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "scheduler_min_lr": 1e-6,

    "global_clip_value": 0.0,

    "log_targetwise_mae": True,
    "error_percentiles": [50, 90, 95, 99],
    "worst_k": 20,
    "log_worst_k_every": 1,
    "save_epoch_reports": True,
    "save_feature_importance": True,
    "feature_importance_split": "test",
    "feature_importance_repeats": 5,
    "feature_importance_seed": 42,

    "save_final_worst_k": True,
    "final_worst_k": 20,
    "include_smiles_in_worstk": True,

    "enable_hard_mining": True,
    "hard_mining_top_k": 300,
    "hard_mining_percent": 0.0,
    "hard_oversample_mult": 5,
    "stage2_epochs": 2,
    "stage2_learning_rate": 2e-4,
    "stage2_optimizer": "adamw",
    "stage2_weight_decay": 1e-4,
    "stage2_loss_type": "huber",
    "stage2_huber_delta": 2.0,
    "stage2_scheduler": "plateau",
    "stage2_scheduler_metric": "mae",
    "stage2_scheduler_patience": 3,
    "stage2_scheduler_factor": 0.5,
    "stage2_scheduler_min_lr": 1e-6,
    "save_train_worstk_best_checkpoint": True,
}
