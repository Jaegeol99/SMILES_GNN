import logging
from typing import Any, Dict, List, Optional

# -------------------------
# Dataset
# -------------------------
CSV_PATH: str = "qm9_valid_smiles_atomization_free_energy.csv"
CSV_SMILES_COL: str = "smiles"
CSV_TARGET_COL: str = "A_G_eV"
CSV_INDEX_COL: str = "i"  # optional; if missing, row number is used

# Use None to load all rows
MAX_SAMPLES: Optional[int] = 10000

# Target metadata (used for reporting/plots)
PROPERTY_NAMES: List[str] = ["QM9_G_free_energy_298K_eV"]
LABEL_COLS: List[str] = ["G"]  # kept for backward-compat; effectively 1 target

# -------------------------
# Outputs
# -------------------------
OUTPUT_DIR: str = "outputs"
MODEL_SAVE_PATH: str = f"{OUTPUT_DIR}/lohc_model.pth"
STAGE2_MODEL_SAVE_PATH: str = f"{OUTPUT_DIR}/lohc_model_stage2.pth"
LABEL_SCALING_PARAMS_PATH: str = f"{OUTPUT_DIR}/label_scaler.npz"
GLOBAL_SCALING_PARAMS_PATH: str = f"{OUTPUT_DIR}/global_desc_standard_scaler.npz"
ABLATION_RESULTS_PATH: str = f"{OUTPUT_DIR}/ablation_results.csv"
REPORT_DIR: str = f"{OUTPUT_DIR}/reports"

# -------------------------
# Logging
# -------------------------
LOGGING_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"

# -------------------------
# Reproducibility
# -------------------------
SEED: int = 42

# -------------------------
# Hyperparameters
# -------------------------
# Keep options minimal; anything not used by train.py is intentionally omitted.
HYPERPARAMS: Dict[str, Any] = {
    # data/split
    "batch_size": 32,
    "epochs": 50,
    "test_split_ratio": 0.2,
    "random_state": 42,

    # model
    "hidden_dim": 128,
    "num_layers": 4,
    "dropout_rate": 0.1,
    "num_output_features": 1,

    # optimization
    "learning_rate": 1e-3,
    "optimizer": "adamw",          # {"adam", "adamw"}
    "weight_decay": 1e-4,          # only used for AdamW

    # loss
    "loss_type": "huber",          # {"mse", "huber"}
    "huber_delta": 1.0,            # Huber transition point in SCALED label units

    # stability
    "grad_clip_norm": 1.0,         # 0 or None disables

    # scheduler
    "scheduler": "plateau",        # {"none", "plateau"}
    "scheduler_metric": "mae",     # {"mae", "loss"} monitored by plateau scheduler
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "scheduler_min_lr": 1e-6,

    # global descriptor scaling
    "global_clip_value": 0.0,      # clip z-scores to [-c, c]; 0 disables

    # diagnostics / reporting
    "log_targetwise_mae": True,
    "error_percentiles": [50, 95, 99],
    "worst_k": 20,
    "log_worst_k_every": 1,        # 1 = every epoch
    "save_epoch_reports": True,

    # final (best-checkpoint) worst-k CSV dumps
    "save_final_worst_k": True,
    "final_worst_k": 20,
    "include_smiles_in_worstk": True,


# hard-example mining + 2-stage fine-tune (optional)
"enable_hard_mining": True,
"hard_mining_top_k": 300,       # 0 disables; otherwise pick top-k hard samples from TRAIN set by abs error
"hard_mining_percent": 0.0,     # used only if top_k==0; e.g. 0.02 = top 2%
"hard_oversample_mult": 5,      # repeat each hard sample this many times in stage-2 train loader
"stage2_epochs": 20,
"stage2_learning_rate": 2e-4,
"stage2_optimizer": "adamw",    # {"adam", "adamw"}
"stage2_weight_decay": 1e-4,
"stage2_loss_type": "huber",    # {"mse", "huber"}
"stage2_huber_delta": 2.0,      # larger -> more penalty on big errors (closer to MSE)
"stage2_scheduler": "plateau",  # {"none", "plateau"}
"stage2_scheduler_metric": "mae",
"stage2_scheduler_patience": 3,
"stage2_scheduler_factor": 0.5,
"stage2_scheduler_min_lr": 1e-6,
"save_train_worstk_best_checkpoint": True,

    # ablation (run with: python train.py --ablation)
    "run_ablation": False,
    "ablation_epochs": 30,
    "ablation_repeats": 1,
    "ablation_huber_deltas": [0.5, 1.0, 2.0],
    "ablation_grad_clips": [0.0, 1.0, 2.0],
    "ablation_schedulers": ["none", "plateau"],
}
