import os
import sys
import random
import numpy as np
import pandas as pd
import torch

# GUI 창을 열지 않고 백그라운드에서 이미지만 생성하도록 강제 설정
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Draw import SimilarityMaps

import shap
from captum.attr import IntegratedGradients
from torch_geometric.data import Batch

# 기존 프로젝트 모듈 임포트
from config import HYPERPARAMS, SEED
from data_processing import (
    load_and_preprocess_qm9_data, apply_global_scaling, 
    inverse_scale_labels, smiles_to_graph_data
)
from train import build_model, set_seed

# =========================================================================
# 물성별 모델 및 스케일링 파일 경로 적용
# =========================================================================
PROPERTY_MODELS = {
    "Gibbs": {
        "model_path": "outputs/lohc_model.pth",
        "global_npz": "outputs/global_desc_standard_scaler.npz",
        "label_npz": "outputs/label_scaler.npz"
    },
    "MP": {
        "model_path": "outputs/lohc_model_mp.pth",
        "global_npz": "outputs/global_desc_standard_scaler_mp.npz",
        "label_npz": "outputs/label_scaler_mp.npz"
    },
    "BP": {
        "model_path": "outputs/lohc_model_bp.pth",
        "global_npz": "outputs/global_desc_standard_scaler_bp.npz",
        "label_npz": "outputs/label_scaler_bp.npz"
    }
}

SCREENING_CSV_PATH = "screening_results/Stage4_Final_Passed_MP.csv"

def load_trained_model(model_path, device, num_node_features):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 가중치 파일을 찾을 수 없습니다: {model_path}")
    
    model = build_model(num_node_features, HYPERPARAMS, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def convert_smiles_to_graph(smiles):
    result = smiles_to_graph_data(smiles, labels=[0.0])
    if result is None:
        raise ValueError(f"SMILES 변환 실패: {smiles}")
    return result[0], result[1]

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs("outputs", exist_ok=True)

    # 1. SHAP Background 데이터를 만들기 위해 기존 학습 데이터 일부 로드
    print("SHAP 분석용 Background 데이터 준비 중... (기존 데이터 로딩)")
    data_list, num_node_features = load_and_preprocess_qm9_data()
    
    indices = list(range(len(data_list)))
    train_indices, _ = train_test_split(
        indices, test_size=float(HYPERPARAMS["test_split_ratio"]), random_state=int(HYPERPARAMS["random_state"])
    )

    desc_names = [desc[0] for desc in Descriptors.descList]
    bg_sample_indices = random.sample(train_indices, min(20, len(train_indices)))
    
    # 2. 스크리닝 결과 CSV 파일 읽기
    if not os.path.exists(SCREENING_CSV_PATH):
        print(f"[오류] 스크리닝 결과 파일을 찾을 수 없습니다: {SCREENING_CSV_PATH}")
        return
    df = pd.read_csv(SCREENING_CSV_PATH)
    print(f"\n총 {len(df)}개의 분자에 대해 분석을 시작합니다.")

    # =========================================================================
    # 3. 물성(Gibbs, MP, BP)별로 순회하며 분석 진행
    # =========================================================================
    for prop_name, paths in PROPERTY_MODELS.items():
        print(f"\n{'='*60}")
        print(f"🚀 [{prop_name}] 물성에 대한 기여도 분석 시작...")
        print(f"{'='*60}")

        model_path = paths["model_path"]
        global_npz_path = paths["global_npz"]
        label_npz_path = paths["label_npz"]

        if not os.path.exists(model_path):
            print(f"[경고] {prop_name} 모델 파일이 없어 건너뜁니다: {model_path}")
            continue

        # 해당 물성의 모델 로드
        model = load_trained_model(model_path, device, num_node_features)

        # 해당 물성의 글로벌 스케일링 파라미터 로드
        if os.path.exists(global_npz_path):
            npz_g = np.load(global_npz_path)
            g_mean, g_std = npz_g['mean'], npz_g['std']
        else:
            print(f"[경고] {global_npz_path} 파일이 없습니다. 스케일링 없이 진행합니다.")
            g_mean, g_std = None, None

        # 해당 물성의 라벨 스케일링 파라미터 로드
        if os.path.exists(label_npz_path):
            npz_lbl = np.load(label_npz_path)
            lbl_median, lbl_iqr = npz_lbl['median'], npz_lbl['iqr']
        else:
            lbl_median, lbl_iqr = None, None

        # SHAP Background 글로벌 특성 추출 (해당 물성의 스케일링 적용)
        bg_global_features =[]
        for idx in bg_sample_indices:
            atom_d, line_d = data_list[idx]
            g_feat = atom_d.g.clone()
            if g_mean is not None and g_std is not None:
                g_feat = (g_feat - torch.tensor(g_mean, dtype=torch.float)) / torch.tensor(g_std, dtype=torch.float)
            bg_global_features.append(g_feat.numpy()[0])
        bg_global_features = np.array(bg_global_features)

        # 각 분자(Row)별 분석
        for idx, row in df.iterrows():
            dehydro_smiles = row.get('Dehydro_SMILES', None)
            hydro_smiles = row.get('Hydro_SMILES', None)
            
            row_id = row.get('ID', f"row_{idx}")
            if pd.isna(row_id) or str(row_id).strip() == "":
                row_id = f"row_{idx}"

            for state_name, target_smiles in[("Dehydro", dehydro_smiles), ("Hydro", hydro_smiles)]:
                if pd.isna(target_smiles) or not isinstance(target_smiles, str):
                    continue

                print(f"  ▶ [{prop_name}] ID: {row_id} | 상태: {state_name}")

                try:
                    atom_data, line_data = convert_smiles_to_graph(target_smiles)

                    # 글로벌 스케일링 적용
                    if g_mean is not None and g_std is not None:
                        clip_value = float(HYPERPARAMS.get("global_clip_value", 0.0))
                        apply_global_scaling([(atom_data, line_data)], g_mean, g_std, clip_value=clip_value if clip_value > 0 else None)

                    t_atom_batch = Batch.from_data_list([atom_data]).to(device)
                    t_line_batch = Batch.from_data_list([line_data]).to(device)

                    # ---------------------------------------------------------
                    #[분석 1] SHAP Waterfall Plot
                    # ---------------------------------------------------------
                    def shap_predict_wrapper(global_features_numpy):
                        preds =[]
                        with torch.no_grad():
                            for g_np in global_features_numpy:
                                temp_atom = t_atom_batch.clone()
                                temp_atom.g = torch.tensor(g_np, dtype=torch.float32).unsqueeze(0).to(device)
                                pred, _ = model(temp_atom, t_line_batch)
                                pred_val = pred.cpu().numpy()
                                
                                if lbl_median is not None and lbl_iqr is not None:
                                    pred_val = inverse_scale_labels(pred_val, lbl_median, lbl_iqr)
                                preds.append(pred_val[0])
                        return np.array(preds)

                    explainer = shap.KernelExplainer(shap_predict_wrapper, bg_global_features)
                    target_g_feat = t_atom_batch.g.cpu().numpy()
                    shap_values = explainer.shap_values(target_g_feat, nsamples=500)
                    
                    exp_val = float(explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
                    sv_1d = np.array(shap_values[0] if isinstance(shap_values, list) else shap_values).flatten()
                    feat_1d_scaled = np.array(target_g_feat).flatten()

                    if g_mean is not None and g_std is not None:
                        feat_1d_raw = feat_1d_scaled * g_std + g_mean
                    else:
                        feat_1d_raw = feat_1d_scaled

                    exp = shap.Explanation(values=sv_1d, base_values=exp_val, data=feat_1d_raw, feature_names=desc_names)

                    plt.figure(figsize=(10, 6))
                    shap.plots.waterfall(exp, max_display=10, show=False)
                    
                    shap_filename = f"outputs/shap_{row_id}_{state_name}_{prop_name}.png"
                    plt.savefig(shap_filename, bbox_inches='tight', dpi=300)
                    plt.close('all') # [수정] 확실한 메모리 해제

                    # ---------------------------------------------------------
                    # [분석 2] 원자 단위 Attribution (Integrated Gradients)
                    # ---------------------------------------------------------
                    input_x_3d = t_atom_batch.x.clone().unsqueeze(0).requires_grad_(True)

                    def ig_forward_wrapper(node_features_3d):
                        B = node_features_3d.shape[0]
                        node_features_2d = node_features_3d.view(-1, node_features_3d.shape[-1])
                        
                        atom_list =[atom_data.clone() for _ in range(B)]
                        line_list =[line_data.clone() for _ in range(B)]
                        
                        temp_atom_batch = Batch.from_data_list(atom_list).to(device)
                        temp_line_batch = Batch.from_data_list(line_list).to(device)
                        
                        temp_atom_batch.x = node_features_2d
                        pred, _ = model(temp_atom_batch, temp_line_batch)
                        return pred

                    ig = IntegratedGradients(ig_forward_wrapper)
                    
                    # [수정] n_steps를 20으로 줄여 메모리 초과(OOM) 방지
                    attributions, delta = ig.attribute(input_x_3d, target=0, n_steps=20, return_convergence_delta=True)
                    attributions = attributions.squeeze(0)
                    atom_attributions = attributions.sum(dim=1).cpu().detach().numpy()
                    
                    mol = Chem.MolFromSmiles(target_smiles)
                    if mol is not None:
                        # [수정] 1. 2D 좌표 명시적 생성 (Segfault 방지)
                        AllChem.Compute2DCoords(mol)
                        
                        # [수정] 2. NaN 및 Inf 값 안전하게 0으로 치환 (Segfault 방지)
                        atom_attributions = np.nan_to_num(atom_attributions, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        max_abs = np.max(np.abs(atom_attributions))
                        if max_abs > 0:
                            atom_attributions = atom_attributions / max_abs

                        try:
                            # [수정] 3. contourLines=0 으로 설정하여 Matplotlib Triangulation 충돌 방지
                            fig = SimilarityMaps.GetSimilarityMapFromWeights(
                                mol, atom_attributions.tolist(), colorMap='bwr', contourLines=0, alpha=0.5, size=(400, 400)
                            )
                            
                            atom_filename = f"outputs/atom_attr_{row_id}_{state_name}_{prop_name}.png"
                            fig.savefig(atom_filename, bbox_inches='tight', dpi=300)
                            plt.close('all') # [수정] 확실한 메모리 해제
                        except Exception as e:
                            print(f"    ->[오류] 원자 기여도 시각화 실패: {e}")
                            plt.close('all')
                    else:
                        print("    ->[오류] RDKit 분자 객체 생성 실패")

                except Exception as e:
                    print(f"    ->[오류] 분석 중 문제 발생: {e}")
                
                # [수정] 로그가 버퍼에 갇히지 않고 즉시 출력되도록 강제
                sys.stdout.flush()

    print("\n모든 물성 및 분자에 대한 분석이 완료되었습니다!")

if __name__ == "__main__":
    main()