import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem  # 3D 구조 생성을 위해 필요
from torch_geometric.loader import DataLoader
import traceback
import time
import os
import logging
from typing import Optional, List, Tuple

# --- 필요한 사용자 정의 모듈 임포트 ---
try:
    from data_processing import smiles_to_graph_data, initialize_feature_scaling, inverse_scale_labels
    from gnn_model import LOHCGNN
    from config import (
        HYPERPARAMS, PROPERTY_NAMES, LABEL_COLS, MODEL_SAVE_PATH, LABEL_SCALING_PARAMS_PATH
    )
    from feature_configs import (
        TOTAL_FEATURE_DIMENSION, NUM_BOND_FEATURES, NUM_LINE_EDGE_FEATURES
    )
except ImportError as e:
    print(f"Error importing required modules or config constants: {e}")
    print("Ensure data_processing.py, gnn_model.py, config.py, and feature_configs.py are accessible.")
    exit(1)

# --- Configuration ---
CONFIG = {
    'input_smiles_file': 'candidates.smi',  # 입력 파일명 확인 필요
    'trained_model_path': MODEL_SAVE_PATH,
    'label_scaling_params_path': LABEL_SCALING_PARAMS_PATH,
    'output_results_file': 'promising_lohc_candidates.xlsx',
    'batch_size': HYPERPARAMS.get('batch_size', 32),
    'num_workers': 0,
    'max_potential': 0.12,
    'min_potential': 0.0,
    'min_capacity': 5.5,
}

# --- Setup Logging ---
log_level = os.environ.get('LOGGING_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO),
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def load_label_scaling_params(filepath):
    try:
        data = np.load(filepath)
        if 'min_vals' in data and 'max_vals' in data:
            logging.info(f"Loaded label scaling parameters from {filepath}")
            return data['min_vals'], data['max_vals']
        else:
            logging.error(f"Error: 'min_vals' or 'max_vals' not found in {filepath}")
            return None, None
    except FileNotFoundError:
        logging.error(f"Error: Label scaling parameter file not found at {filepath}")
        return None, None
    except Exception as e:
        logging.error(f"Error loading label scaling parameters: {e}")
        return None, None

def is_physically_valid(mol: Chem.Mol, check_3d: bool = True) -> bool:
    """
    분자가 물리/화학적으로 타당한지 검사합니다.
    1. Sanitization
    2. 전하(Charge) 체크 (중성만 허용)
    3. 라디칼 체크
    4. 3D 구조 생성 가능성 체크 (Ring Strain 등 확인)
    """
    if mol is None:
        return False
    
    try:
        # 1. Sanitize
        Chem.SanitizeMol(mol)
        
        # 2. 전하 체크: LOHC는 중성 분자여야 함
        if Chem.GetFormalCharge(mol) != 0:
            return False
            
        # 3. 라디칼 체크
        for atom in mol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                return False
        
        # 4. 3D 구조 생성 체크 (가장 중요: 물리적으로 불가능한 구조 필터링)
        if check_3d:
            mol_h = Chem.AddHs(mol) # 수소를 붙여서 정확한 입체 장애 확인
            # EmbedMolecule이 -1을 반환하면 3D 구조 생성 실패 (불가능한 구조)
            res = AllChem.EmbedMolecule(mol_h, randomSeed=42, maxAttempts=50)
            if res == -1:
                return False
                
        return True
    except Exception:
        return False

def hydrogenate_smiles_conservative(dehydro_smiles: str) -> Optional[str]:
    """
    불포화 결합을 단일 결합으로 변환하여 수소화된 SMILES를 생성합니다.
    생성 후 물리적 타당성(3D 구조 등)을 검증합니다.
    """
    mol = Chem.MolFromSmiles(dehydro_smiles, sanitize=False)
    if mol is None:
        return None

    try:
        mol.UpdatePropertyCache(strict=False)
        rw_mol = Chem.RWMol(mol)

        # 모든 결합을 순회하며 단일 결합으로 변경
        # (방향족성 제거 및 이중/삼중 결합 제거)
        for bond in rw_mol.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
        
        # 원자의 방향족성 플래그 제거
        for atom in rw_mol.GetAtoms():
            atom.SetIsAromatic(False)
            
        saturated_mol = rw_mol.GetMol()
        
        # 생성된 분자가 물리적으로 가능한지 검증 (3D 체크 포함)
        if not is_physically_valid(saturated_mol, check_3d=True):
            return None
            
        return Chem.MolToSmiles(saturated_mol, canonical=True)

    except Exception as e:
        logging.debug(f"Hydrogenation failed for {dehydro_smiles}: {e}")
        return None

# --- Main Screening Logic ---
def screen_candidates():
    logging.info("Starting LOHC candidate screening with Enhanced Validation...")
    start_time = time.time()

    # 1. Load Scaling Parameters
    label_min_values, label_max_values = load_label_scaling_params(CONFIG['label_scaling_params_path'])
    if label_min_values is None or label_max_values is None:
        logging.error("Could not load label scaling parameters. Please run train.py first.")
        return

    label_range = label_max_values - label_min_values
    label_range[np.abs(label_range) < 1e-9] = 1.0

    # 2. Load SMILES
    logging.info(f"Loading SMILES from {CONFIG['input_smiles_file']}...")
    try:
        candidate_smiles_with_id = []
        with open(CONFIG['input_smiles_file'], 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    candidate_id, smiles_str = parts[0], parts[1]
                    if smiles_str:
                        candidate_smiles_with_id.append((candidate_id, smiles_str))
                elif len(parts) == 1 and parts[0]:
                     candidate_smiles_with_id.append((str(line_num + 1), parts[0]))
        logging.info(f"Loaded {len(candidate_smiles_with_id)} candidate SMILES.")
    except Exception as e:
        logging.error(f"Error reading SMILES file: {e}")
        return

    # 3. Initialize Feature Scaling
    try:
        initialize_feature_scaling(TOTAL_FEATURE_DIMENSION)
        logging.info(f"Feature scaling initialized. Dim: {TOTAL_FEATURE_DIMENSION}")
    except Exception as e:
         logging.error(f"Feature scaling setup failed: {e}")
         return

    # 4. Generate Pairs & Graph Data
    logging.info("Generating valid molecule pairs and graph data...")
    paired_graph_data_list = []
    valid_smiles_pairs_with_id = []
    processed_count = 0
    skipped_count = 0
    
    # Dummy labels for prediction input
    dummy_labels_original = [0.0] * len(LABEL_COLS)
    dummy_labels_scaled = ((np.array(dummy_labels_original) - label_min_values) / label_range).tolist()

    for i, (candidate_id, smiles_de) in enumerate(candidate_smiles_with_id):
        if (i + 1) % 1000 == 0:
            logging.info(f"  Processed {i+1}/{len(candidate_smiles_with_id)}...")
            
        if not smiles_de:
            skipped_count += 1
            continue
        
        # [Step 1] Dehydrogenated 분자 검증 (전하, 라디칼 등)
        mol_de = Chem.MolFromSmiles(smiles_de)
        if not is_physically_valid(mol_de, check_3d=False): # Dehydro는 3D 체크까지는 안 해도 됨 (선택사항)
            # logging.debug(f"Skipping {candidate_id}: Invalid Dehydro structure (Charge/Radical).")
            skipped_count += 1
            continue

        # [Step 2] Hydrogenation & Validation (3D Check 포함)
        smiles_hy = hydrogenate_smiles_conservative(smiles_de)
        
        if not smiles_hy:
            # logging.debug(f"Skipping {candidate_id}: Hydrogenation failed or physically impossible.")
            skipped_count += 1
            continue
            
        # [Step 3] Graph Conversion
        graph_pair_de = smiles_to_graph_data(smiles_de, dummy_labels_scaled)
        graph_pair_hy = smiles_to_graph_data(smiles_hy, dummy_labels_scaled)
        
        if graph_pair_de and graph_pair_hy:
            paired_graph_data_list.append((graph_pair_de, graph_pair_hy))
            valid_smiles_pairs_with_id.append({
                'ID': candidate_id, 
                'Dehydrogenated_SMILES': smiles_de, 
                'Hydrogenated_SMILES': smiles_hy
            })
            processed_count += 1
        else:
            skipped_count += 1

    logging.info(f"Generated {processed_count} valid graph pairs. Skipped {skipped_count} (Invalid/Impossible structures).")
    
    if not paired_graph_data_list:
        logging.error("No valid candidates remaining after validation. Exiting.")
        return

    # 5. Load Model
    logging.info(f"Loading model from {CONFIG['trained_model_path']}...")
    try:
        model = LOHCGNN(
            node_in_dim=TOTAL_FEATURE_DIMENSION,
            edge_in_dim=NUM_BOND_FEATURES,
            line_edge_in_dim=NUM_LINE_EDGE_FEATURES,
            hidden_dim=HYPERPARAMS['hidden_dim'],
            num_layers=HYPERPARAMS['num_layers'],
            num_output_features=HYPERPARAMS['num_output_features'],
            dropout_rate=HYPERPARAMS['dropout_rate']
        )
        model.load_state_dict(torch.load(CONFIG['trained_model_path'], map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    # 6. Prediction
    logging.info("Predicting properties...")
    all_predictions_scaled = []
    data_loader = DataLoader(paired_graph_data_list, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        for data_pair in data_loader:
            (atom_de_b, line_de_b), (atom_hy_b, line_hy_b) = data_pair
            
            # Move to device
            atom_batch = atom_hy_b.to(device)
            atom_batch.x_de = atom_de_b.x.to(device)
            atom_batch.edge_index_de = atom_de_b.edge_index.to(device)
            atom_batch.edge_attr_de = atom_de_b.edge_attr.to(device)
            atom_batch.batch_de = atom_de_b.batch.to(device)
            
            line_batch = line_hy_b.to(device)
            line_batch.x_de = line_de_b.x.to(device)
            line_batch.edge_index_de = line_de_b.edge_index.to(device)
            line_batch.edge_attr_de = line_de_b.edge_attr.to(device)
            line_batch.batch_de = line_de_b.batch.to(device)
            
            try:
                output_scaled = model(atom_batch, line_batch)
                all_predictions_scaled.append(output_scaled.cpu().numpy())
            except Exception as e:
                logging.error(f"Prediction error: {e}")
                # Error handling: fill with NaNs
                num_graphs = atom_batch.num_graphs
                all_predictions_scaled.append(np.full((num_graphs, len(LABEL_COLS)), np.nan))

    if not all_predictions_scaled:
        logging.error("No predictions generated.")
        return

    # 7. Post-processing & Filtering
    predictions_scaled_np = np.concatenate(all_predictions_scaled, axis=0)
    predictions_original_np = inverse_scale_labels(predictions_scaled_np, label_min_values, label_max_values)

    results_df = pd.DataFrame(valid_smiles_pairs_with_id)
    for i, name in enumerate(PROPERTY_NAMES):
        results_df[f'Predicted_{name}'] = predictions_original_np[:, i]

    # Filter: Remove NaNs
    pred_cols = [f'Predicted_{name}' for name in PROPERTY_NAMES]
    results_df = results_df.dropna(subset=pred_cols)

    # Filter: Potential
    potential_col = f'Predicted_{PROPERTY_NAMES[2]}' # Standard oxidation potential
    potential_mask = (results_df[potential_col] <= CONFIG['max_potential']) & \
                     (results_df[potential_col] >= CONFIG['min_potential'])
    filtered_df = results_df[potential_mask]
    logging.info(f"Candidates after Potential filter ({CONFIG['min_potential']}~{CONFIG['max_potential']} V): {len(filtered_df)}")

    # Filter: Capacity
    capacity_col = f'Predicted_{PROPERTY_NAMES[3]}' # Capacity
    capacity_mask = filtered_df[capacity_col] >= CONFIG['min_capacity']
    filtered_df = filtered_df[capacity_mask]
    logging.info(f"Candidates after Capacity filter (>= {CONFIG['min_capacity']} wt%): {len(filtered_df)}")

    # 8. Save Results
    if not filtered_df.empty:
        try:
            sorted_df = filtered_df.sort_values(by=[capacity_col, potential_col], ascending=[False, True])
            sorted_df.to_excel(CONFIG['output_results_file'], index=False, engine='openpyxl')
            logging.info(f"Saved {len(sorted_df)} candidates to {CONFIG['output_results_file']}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")
    else:
        logging.info("No candidates passed all filters.")

    logging.info(f"Screening completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    screen_candidates()