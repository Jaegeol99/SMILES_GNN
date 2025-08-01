# config.py

import os
import logging
from typing import List, Dict, Any

# 1. 데이터 관련 설정
DATA_FILE_PATH: str = 'Charge_data.xlsx'  # 사용할 파일 이름
SMILES_COL: str = 'SMILES'                         # SMILES 정보가 있는 컬럼
PROPERTY_NAMES: List[str] = ['Atomic Charge']      # 예측할 속성 이름

# 2. 모델 및 스케일러 저장 경로
MODEL_SAVE_PATH: str = 'atomic_charge_model.pth'
LABEL_SCALING_PARAMS_PATH: str = 'atomic_charge_scaler.npz'

# 3. 로깅 설정
LOGGING_LEVEL: int = logging.INFO
LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'

# 4. 하이퍼파라미터
HYPERPARAMS: Dict[str, Any] = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 150,
    'hidden_dim': 128,
    'num_layers': 4,
    'dropout_rate': 0.5,
    'test_split_ratio': 0.2,
    'random_state': 42,
    'num_output_features': 1  # 예측할 타겟은 '원자 전하' 하나입니다.
}