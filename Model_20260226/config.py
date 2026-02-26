import logging
from typing import Any, Dict, List, Optional

# -------------------------
# Dataset
# -------------------------
CSV_PATH: str = "qm9.csv"
CSV_SMILES_COL: str = "smiles"
CSV_TARGET_COL: str = "A_G_eV"
CSV_INDEX_COL: str = "i"  # optional; if missing, row number is used

# Use None to load all rows
MAX_SAMPLES: Optional[int] = None  # 전체 데이터를 학습하기 위해 None으로 변경

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
    "batch_size": 50,          # D-MPNN 표준 배치 사이즈
    "epochs": 200,             # 전체 데이터 수렴을 위한 에포크 상향
    "test_split_ratio": 0.2,
    "random_state": 42,

    # model
    "hidden_dim": 300,         # 모델 수용력을 높이기 위해 300으로 확장
    "num_layers": 6,           # 분자 전체 반경 커버를 위해 깊이 증가
    "dropout_rate": 0.0,       # 대규모 데이터이므로 드롭아웃 미사용
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
    "scheduler": "exponential",    # 초정밀 탐색을 위해 지수 감쇠 스케줄러 도입
    "scheduler_gamma": 0.95,       # 매 에포크마다 학습률 5% 감소
    "scheduler_metric": "mae",     # {"mae", "loss"} monitored by plateau scheduler
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "scheduler_min_lr": 1e-6,

    # global descriptor scaling
    "global_clip_value": 0.0,      # clip z-scores to [-c, c]; 0 disables

    # diagnostics / reporting
    "log_targetwise_mae": True,
    "error_percentiles": [50, 90, 95, 99], # [수정] 누락된 리스트 값 추가
    "worst_k": 20,
    "log_worst_k_every": 1,        # 1 = every epoch
    "save_epoch_reports": True,

    # final (best-checkpoint) worst-k CSV dumps
    "save_final_worst_k": True,
    "final_worst_k": 20,
    "include_smiles_in_worstk": True,

    # hard-example mining + 2-stage fine-tune (optional)
    "enable_hard_mining": True,
    "hard_mining_top_k": 300,       
    "hard_mining_percent": 0.0,     
    "hard_oversample_mult": 5,      
    "stage2_epochs": 20,
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

    # ablation (run with: python train.py --ablation)
    "run_ablation": False,
    "ablation_epochs": 30,
    "ablation_repeats": 1,
    "ablation_huber_deltas": [0.5, 1.0, 2.0],
    "ablation_grad_clips": [0.0, 1.0, 2.0],
    "ablation_schedulers": ["none", "plateau"],
}