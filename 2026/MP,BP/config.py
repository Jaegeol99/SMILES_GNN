# --- START OF FILE config.py ---

import logging
from typing import Any, Dict, List, Optional

# -------------------------
# Dataset
# -------------------------
CSV_PATH: str = "mp.csv"
CSV_SMILES_COL: str = "smiles"
CSV_TARGET_COL: str = "mp"
CSV_INDEX_COL: str = "i"  

MAX_SAMPLES: Optional[int] = None  

PROPERTY_NAMES: List[str] = ["Melting Point"]
LABEL_COLS: List[str] = ["mp"]  

# -------------------------
# Outputs
# -------------------------
OUTPUT_DIR: str = "outputs"
MODEL_SAVE_PATH: str = f"{OUTPUT_DIR}/lohc_model_mp.pth"
STAGE2_MODEL_SAVE_PATH: str = f"{OUTPUT_DIR}/lohc_model_stage2_mp.pth"
LABEL_SCALING_PARAMS_PATH: str = f"{OUTPUT_DIR}/label_scaler_mp.npz"
GLOBAL_SCALING_PARAMS_PATH: str = f"{OUTPUT_DIR}/global_desc_standard_scaler_mp.npz"
ABLATION_RESULTS_PATH: str = f"{OUTPUT_DIR}/ablation_line_graph_results.csv"
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
HYPERPARAMS: Dict[str, Any] = {
    "model_type": "lohcgnn",
    
    "use_line_edge_features": True, # 제안하는 핵심 피처 (가상 각도, 고리 긴장, 공액 흐름)

    # data/split
    "batch_size": 50,          
    "epochs": 200,             
    "test_split_ratio": 0.2,
    "random_state": 42,

    # model
    "hidden_dim": 300,         
    "num_layers": 6,           
    "dropout_rate": 0.0,       
    "num_output_features": 1,

    # optimization
    "learning_rate": 1e-3,
    "optimizer": "adamw",          
    "weight_decay": 1e-4,          

    # loss
    "loss_type": "huber",          
    "huber_delta": 1.0,            

    # stability
    "grad_clip_norm": 1.0,         

    # scheduler
    "scheduler": "exponential",    
    "scheduler_gamma": 0.95,       
    "scheduler_metric": "mae",     
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "scheduler_min_lr": 1e-6,

    # global descriptor scaling
    "global_clip_value": 0.0,      

    # diagnostics / reporting
    "log_targetwise_mae": True,
    "error_percentiles": [50, 90, 95, 99], 
    "worst_k": 20,
    "log_worst_k_every": 1,        
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

    # =================================================================
    # 일반 학습 모드 (Ablation 끄기)
    # =================================================================
    "run_ablation": False, 
    "ablation_epochs": 200, 
    "ablation_repeats": 3, 
    "ablation_configs": []
}