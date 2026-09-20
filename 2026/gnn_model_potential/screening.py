import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from torch_geometric.loader import DataLoader
import time
import os
import logging
import gc
from tqdm import tqdm

# --- RDKit 경고 로그 끄기 ---
RDLogger.DisableLog('rdApp.*')

try:
    from data_processing import smiles_to_graph_data
    from gnn_model import LOHCGNN
    from config import HYPERPARAMS
    from feature_configs import TOTAL_FEATURE_DIMENSION, NUM_BOND_FEATURES, NUM_LINE_EDGE_FEATURES, LINE_NODE_FEATURE_DIM, GLOBAL_FEATURE_DIM
except ImportError as e:
    print(f"Error importing required modules: {e}")
    exit(1)

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
OUTPUT_DIR = 'screening_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = {
    'input_csv_file': 'lohc_all_pairs.csv',
    
    'out_stage1_capacity': os.path.join(OUTPUT_DIR, 'Stage1_Passed_Capacity.csv'),
    'out_stage2_potential': os.path.join(OUTPUT_DIR, 'Stage2_Passed_Potential.csv'),
    'out_stage3_bp': os.path.join(OUTPUT_DIR, 'Stage3_Passed_BP.csv'),
    'out_stage4_final': os.path.join(OUTPUT_DIR, 'Stage4_Final_Passed_MP.csv'), 
    
    'batch_size': HYPERPARAMS.get('batch_size', 32),
    'chunk_size': 20000,  # 메모리 관리를 위한 청크 사이즈 설정
    
    'global_scaler_path': 'outputs/global_desc_standard_scaler.npz',
    'gibbs_model_path': 'outputs/lohc_model.pth',
    'gibbs_scaler_path': 'outputs/label_scaler.npz',
    'mp_model_path': 'outputs/lohc_model_mp.pth',
    'mp_scaler_path': 'outputs/label_scaler_mp.npz',
    'bp_model_path': 'outputs/lohc_model_bp.pth',
    'bp_scaler_path': 'outputs/label_scaler_bp.npz',

    'G_H2_ATOMIZATION_eV': -4.234926, 

    'min_capacity': 5.5,       
    'min_potential': 0.0,      
    'max_potential': 1.23,     
    'min_boiling_point': 85.0,
    'max_melting_point': 0.0  
}

# 실시간 로그 확인을 위한 설정
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# ==========================================
# 2. 유틸리티 함수
# ==========================================
def load_robust_scaler(filepath):
    data = np.load(filepath)
    return data['median'], data['iqr']

def inverse_scale(scaled_data, median, iqr):
    return scaled_data * iqr + median

def predict_property(model, data_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for atom_batch, line_batch in data_loader:
            atom_batch, line_batch = atom_batch.to(device), line_batch.to(device)
            preds.append(model(atom_batch, line_batch)[0].cpu().numpy())
    return np.concatenate(preds, axis=0)

def load_model(path, device, pooling_type="add"):
    model = LOHCGNN(
        node_in_dim=TOTAL_FEATURE_DIMENSION, edge_in_dim=NUM_BOND_FEATURES,
        line_node_in_dim=LINE_NODE_FEATURE_DIM, line_edge_in_dim=NUM_LINE_EDGE_FEATURES,
        hidden_dim=HYPERPARAMS['hidden_dim'], num_layers=HYPERPARAMS['num_layers'],
        num_output_features=HYPERPARAMS['num_output_features'], dropout_rate=HYPERPARAMS['dropout_rate'],
        global_in_dim=GLOBAL_FEATURE_DIM,
        pooling_type=pooling_type
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# [수정됨] 수소 저장 용량 계산 로직 수정 (수소 원자 개수 기반 질량 계산)
def calculate_capacity(row):
    s_hy = row['Hydro_SMILES']
    n_h_atoms = float(row['Added_H'])  # Added_H는 수소 원자의 개수
    
    mol_hy = Chem.MolFromSmiles(s_hy)
    if mol_hy is None:
        return 0.0
        
    mw_hy = Descriptors.MolWt(mol_hy)
    
    # 수소 원자 1개의 질량은 약 1.00794 g/mol
    mass_h_released = n_h_atoms * 1.00794 
    
    return (mass_h_released / mw_hy) * 100 if mw_hy > 0 else 0.0

# ==========================================
# 3. 메인 스크리닝 로직
# ==========================================
def unified_screening():
    logging.info("Starting Fast LOHC Screening Pipeline with Chunking...")
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # ---------------------------------------------------------
    # [Step 0] 데이터 로드
    # ---------------------------------------------------------
    try:
        df = pd.read_csv(CONFIG['input_csv_file'])
        df = df[df['Added_H'] > 0].reset_index(drop=True)
        initial_count = len(df)
        logging.info(f"Loaded {initial_count} valid candidates from {CONFIG['input_csv_file']}.")
    except Exception as e:
        logging.error(f"Error reading input file: {e}")
        return

    # ---------------------------------------------------------
    # [Stage 1] 수소 저장 용량(Capacity) 계산 및 1차 필터링
    # ---------------------------------------------------------
    logging.info("\n--- [Stage 1] Fast Capacity Filtering ---")
    tqdm.pandas(desc="Calculating Capacity")
    df['Calculated_Capacity'] = df.progress_apply(calculate_capacity, axis=1)
    
    mask_s1 = df['Calculated_Capacity'] >= CONFIG['min_capacity']
    df_s1 = df[mask_s1].copy().reset_index(drop=True)
    df_s1.to_csv(CONFIG['out_stage1_capacity'], index=False)
    
    logging.info(f"Stage 1 Passed (Capacity >= {CONFIG['min_capacity']} wt%): {len(df_s1)} / {initial_count} pairs saved.")
    if df_s1.empty: return

    # ---------------------------------------------------------
    # [준비] 모델 및 스케일러 미리 로드 (청크 반복 시 시간 절약)
    # ---------------------------------------------------------
    logging.info("\nLoading Models and Scalers...")
    try:
        g_scaler_data = np.load(CONFIG['global_scaler_path'])
        g_mean, g_std = g_scaler_data['mean'], g_scaler_data['std']
        g_std = np.where(g_std < 1e-12, 1.0, g_std)
        g_mean_t = torch.tensor(g_mean, dtype=torch.float)
        g_std_t = torch.tensor(g_std, dtype=torch.float)

        gibbs_med, gibbs_iqr = load_robust_scaler(CONFIG['gibbs_scaler_path'])
        bp_med, bp_iqr = load_robust_scaler(CONFIG['bp_scaler_path'])
        mp_med, mp_iqr = load_robust_scaler(CONFIG['mp_scaler_path'])

        gibbs_model = load_model(CONFIG['gibbs_model_path'], device, pooling_type="add")
        bp_model = load_model(CONFIG['bp_model_path'], device, pooling_type="mean")
        mp_model = load_model(CONFIG['mp_model_path'], device, pooling_type="mean")
    except Exception as e:
        logging.error(f"Error loading models/scalers: {e}")
        return

    # ---------------------------------------------------------
    # [Stage 2~4] 청크(Chunk) 단위 처리 (메모리 최적화)
    # ---------------------------------------------------------
    chunk_size = CONFIG['chunk_size']
    num_chunks = (len(df_s1) + chunk_size - 1) // chunk_size
    
    logging.info(f"\nStarting Chunk Processing: Total {len(df_s1)} rows divided into {num_chunks} chunks (Size: {chunk_size})")

    # 결과를 누적할 리스트
    all_stage2_passed = []
    all_stage3_passed = []
    all_stage4_passed = []

    for i in range(num_chunks):
        chunk_start = i * chunk_size
        chunk_end = min((i + 1) * chunk_size, len(df_s1))
        chunk_df = df_s1.iloc[chunk_start:chunk_end].copy().reset_index(drop=True)
        
        logging.info(f"\n[{i+1}/{num_chunks}] Processing Chunk Rows {chunk_start} ~ {chunk_end}...")

        # --- 1. 그래프 생성 (현재 청크만) ---
        de_graphs, hy_graphs = [], []
        valid_indices = []
        dummy_label = [0.0]
        failed_count = 0

        for idx, row in chunk_df.iterrows():
            s_de, s_hy = row['Dehydro_SMILES'], row['Hydro_SMILES']
            try:
                g_de = smiles_to_graph_data(s_de, dummy_label)
                g_hy = smiles_to_graph_data(s_hy, dummy_label)
                
                if g_de and g_hy:
                    g_de[0].g = (g_de[0].g - g_mean_t) / g_std_t
                    g_de[1].g = (g_de[1].g - g_mean_t) / g_std_t
                    g_hy[0].g = (g_hy[0].g - g_mean_t) / g_std_t
                    g_hy[1].g = (g_hy[1].g - g_mean_t) / g_std_t

                    de_graphs.append(g_de)
                    hy_graphs.append(g_hy)
                    valid_indices.append(idx)
                else:
                    failed_count += 1
            except Exception:
                failed_count += 1

        chunk_df = chunk_df.loc[valid_indices].reset_index(drop=True)
        if failed_count > 0:
            logging.warning(f"Chunk {i+1}: Failed to generate graphs for {failed_count} pairs.")
        
        if chunk_df.empty:
            logging.info(f"Chunk {i+1}: No valid graphs generated. Skipping.")
            continue

        # --- 2. Gibbs Energy & Potential 필터링 ---
        de_loader = DataLoader(de_graphs, batch_size=CONFIG['batch_size'], shuffle=False)
        hy_loader = DataLoader(hy_graphs, batch_size=CONFIG['batch_size'], shuffle=False)

        g_de_pred = inverse_scale(predict_property(gibbs_model, de_loader, device), gibbs_med, gibbs_iqr).flatten()
        g_hy_pred = inverse_scale(predict_property(gibbs_model, hy_loader, device), gibbs_med, gibbs_iqr).flatten()
        
        chunk_df['Predicted_Gibbs_Dehydro_eV'] = g_de_pred
        chunk_df['Predicted_Gibbs_Hydro_eV'] = g_hy_pred
        
        n_h_atoms = chunk_df['Added_H'].values.astype(float)
        n_h2_molecules = n_h_atoms / 2.0  
        n_electrons = n_h_atoms           
        
        chunk_df['Predicted_Potential_V'] = (g_de_pred + (n_h2_molecules * CONFIG['G_H2_ATOMIZATION_eV']) - g_hy_pred) / n_electrons

        mask_s2 = (chunk_df['Predicted_Potential_V'] >= CONFIG['min_potential']) & \
                  (chunk_df['Predicted_Potential_V'] <= CONFIG['max_potential'])
        
        chunk_df = chunk_df[mask_s2].reset_index(drop=True)
        de_graphs = [g for g, m in zip(de_graphs, mask_s2) if m]
        hy_graphs = [g for g, m in zip(hy_graphs, mask_s2) if m]
        
        all_stage2_passed.append(chunk_df.copy())
        logging.info(f"Chunk {i+1}: Stage 2 (Potential) Passed -> {len(chunk_df)} pairs")
        
        if chunk_df.empty: continue

        # --- 3. Boiling Point (BP) 필터링 ---
        de_loader = DataLoader(de_graphs, batch_size=CONFIG['batch_size'], shuffle=False)
        hy_loader = DataLoader(hy_graphs, batch_size=CONFIG['batch_size'], shuffle=False)

        chunk_df['Predicted_BP_Dehydro'] = inverse_scale(predict_property(bp_model, de_loader, device), bp_med, bp_iqr).flatten()
        chunk_df['Predicted_BP_Hydro'] = inverse_scale(predict_property(bp_model, hy_loader, device), bp_med, bp_iqr).flatten()

        mask_s3 = (chunk_df['Predicted_BP_Dehydro'] >= CONFIG['min_boiling_point']) & \
                  (chunk_df['Predicted_BP_Hydro'] >= CONFIG['min_boiling_point'])

        chunk_df = chunk_df[mask_s3].reset_index(drop=True)
        de_graphs = [g for g, m in zip(de_graphs, mask_s3) if m]
        hy_graphs = [g for g, m in zip(hy_graphs, mask_s3) if m]

        all_stage3_passed.append(chunk_df.copy())
        logging.info(f"Chunk {i+1}: Stage 3 (BP) Passed -> {len(chunk_df)} pairs")

        if chunk_df.empty: continue

        # --- 4. Melting Point (MP) 필터링 ---
        de_loader = DataLoader(de_graphs, batch_size=CONFIG['batch_size'], shuffle=False)
        hy_loader = DataLoader(hy_graphs, batch_size=CONFIG['batch_size'], shuffle=False)

        chunk_df['Predicted_MP_Dehydro'] = inverse_scale(predict_property(mp_model, de_loader, device), mp_med, mp_iqr).flatten()
        chunk_df['Predicted_MP_Hydro'] = inverse_scale(predict_property(mp_model, hy_loader, device), mp_med, mp_iqr).flatten()

        mask_s4 = (chunk_df['Predicted_MP_Dehydro'] <= CONFIG['max_melting_point']) & \
                  (chunk_df['Predicted_MP_Hydro'] <= CONFIG['max_melting_point'])

        chunk_df = chunk_df[mask_s4].reset_index(drop=True)
        
        all_stage4_passed.append(chunk_df.copy())
        logging.info(f"Chunk {i+1}: Stage 4 (MP) Passed -> {len(chunk_df)} pairs")

        # --- 메모리 강제 해제 (매우 중요) ---
        del de_graphs, hy_graphs, de_loader, hy_loader, chunk_df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # [Step 5] 전체 결과 병합 및 저장
    # ---------------------------------------------------------
    logging.info("\n--- Merging and Saving Final Results ---")
    
    df_s2_final = pd.concat(all_stage2_passed, ignore_index=True) if all_stage2_passed else pd.DataFrame()
    df_s3_final = pd.concat(all_stage3_passed, ignore_index=True) if all_stage3_passed else pd.DataFrame()
    df_s4_final = pd.concat(all_stage4_passed, ignore_index=True) if all_stage4_passed else pd.DataFrame()

    if not df_s2_final.empty: df_s2_final.to_csv(CONFIG['out_stage2_potential'], index=False)
    if not df_s3_final.empty: df_s3_final.to_csv(CONFIG['out_stage3_bp'], index=False)
    
    if not df_s4_final.empty:
        sort_cols = ['Calculated_Capacity', 'Predicted_Potential_V', 'Predicted_MP_Dehydro', 'Predicted_BP_Dehydro']
        asc_order = [False, True, True, False]
        df_s4_final = df_s4_final.sort_values(by=sort_cols, ascending=asc_order)
        df_s4_final.to_csv(CONFIG['out_stage4_final'], index=False)

    print("\n" + "="*50)
    print(" [Fast Screening Summary (Chunked)]")
    print(f" - Initial valid pairs : {initial_count}")
    print(f" - Stage 1 (Capacity)  : {len(df_s1)} passed")
    print(f" - Stage 2 (Potential) : {len(df_s2_final)} passed")
    print(f" - Stage 3 (BP)        : {len(df_s3_final)} passed")
    print(f" - Stage 4 (MP)        : {len(df_s4_final)} passed (Final)")
    print("="*50 + "\n")
    print(f"All intermediate and final results are saved in the '{OUTPUT_DIR}' directory.")
    logging.info(f"Screening completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    unified_screening()