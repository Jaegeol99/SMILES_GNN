import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import Image
import io
import os
import random

# 기존 프로젝트 파일들 import
from config import (
    DATA_FILE_PATH, DEHYDRO_SMILES_COL, HYDRO_SMILES_COL, 
    LABEL_COLS, HYPERPARAMS, MODEL_SAVE_PATH
)
from data_processing import smiles_to_graph_data, TOTAL_FEATURE_DIMENSION
from feature_configs import NUM_BOND_FEATURES, NUM_LINE_EDGE_FEATURES
from gnn_model import LOHCGNN

# ==========================================
# 1. 함수 정의 (Attention 추출 및 시각화)
# ==========================================

def get_attention_weights(model, smiles_de, smiles_hy, device):
    """
    단일 SMILES 쌍에 대해 모델을 실행하고 어텐션 가중치를 반환합니다.
    """
    # 1. 데이터 변환 (레이블은 더미 값)
    dummy_labels = [0.0] * len(LABEL_COLS)
    data_tuple_de = smiles_to_graph_data(smiles_de, dummy_labels)
    data_tuple_hy = smiles_to_graph_data(smiles_hy, dummy_labels)
    
    if data_tuple_de is None or data_tuple_hy is None:
        return None, None

    (atom_de, line_de), (atom_hy, line_hy) = data_tuple_de, data_tuple_hy

    # 2. 배치 수동 구성 (Batch Size = 1)
    # 모델의 forward()는 Batch 객체를 기대하므로 device로 보내면서 구조를 맞춥니다.
    
    # Atom Graph Batch
    atom_batch = atom_hy.to(device)
    atom_batch.x_de = atom_de.x.to(device)
    atom_batch.edge_index_de = atom_de.edge_index.to(device)
    atom_batch.edge_attr_de = atom_de.edge_attr.to(device)
    
    # 배치 인덱스 생성 (모두 0번 그래프)
    atom_batch.batch = torch.zeros(atom_hy.x.size(0), dtype=torch.long).to(device)
    atom_batch.batch_de = torch.zeros(atom_de.x.size(0), dtype=torch.long).to(device)

    # Line Graph Batch
    line_batch = line_hy.to(device)
    line_batch.x_de = line_de.x.to(device)
    line_batch.edge_index_de = line_de.edge_index.to(device)
    line_batch.edge_attr_de = line_de.edge_attr.to(device)
    
    # Line Graph용 배치 인덱스
    line_batch.batch = torch.zeros(line_hy.x.size(0), dtype=torch.long).to(device)
    line_batch.batch_de = torch.zeros(line_de.x.size(0), dtype=torch.long).to(device)

    # 3. 모델 실행 (Evaluation Mode)
    model.eval()
    with torch.no_grad():
        _, attention_dict = model(atom_batch, line_batch)

    # 4. 결과 추출 (Numpy 변환)
    w_hy = attention_dict['hydrogenated_atom_attention'].cpu().numpy().flatten()
    w_de = attention_dict['dehydrogenated_atom_attention'].cpu().numpy().flatten()

    return w_hy, w_de

def draw_molecule_with_attention(smiles, weights, ax, title):
    """
    rdMolDraw2D를 사용하여 고해상도로 어텐션 가중치를 시각화합니다.
    낮은 가중치: 파란색 -> 높은 가중치: 빨간색
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        ax.text(0.5, 0.5, "Invalid SMILES", ha='center')
        return

    # 1. 정규화 (Min-Max)
    w_min, w_max = weights.min(), weights.max()
    if w_max - w_min < 1e-6:
        norm_weights = np.zeros_like(weights)
    else:
        norm_weights = (weights - w_min) / (w_max - w_min)

    # 2. 컬러맵 설정 (bwr: Blue-White-Red)
    cmap = cm.get_cmap('bwr')
    
    # 하이라이트 설정
    highlight_atom_list = []
    highlight_atom_colors = {}
    
    for i in range(mol.GetNumAtoms()):
        if i < len(norm_weights):
            val = norm_weights[i]
            
            # [핵심 수정] numpy array -> tuple 변환 필수!
            # alpha 값(투명도)을 0.8 정도로 주어 원자 기호가 보이게 함
            rgba_color = cmap(val)
            color_tuple = tuple(rgba_color[:3]) # (R, G, B) 튜플로 변환
            
            highlight_atom_list.append(i)
            highlight_atom_colors[i] = color_tuple

    # 3. rdMolDraw2D를 이용한 고품질 그리기
    # 캔버스 생성
    drawer = rdMolDraw2D.MolDraw2DCairo(400, 400) # Cairo가 없으면 MolDraw2DSVG 권장
    
    # 옵션 설정
    dopts = drawer.drawOptions()
    dopts.useBWAtomPalette() # 원자를 흑백으로 표시하여 컬러 하이라이트 강조
    dopts.padding = 0.05
    dopts.legendFontSize = 20
    
    # 그리기 실행
    drawer.DrawMolecule(
        mol, 
        highlightAtoms=highlight_atom_list, 
        highlightAtomColors=highlight_atom_colors
    )
    drawer.FinishDrawing()
    
    # 4. Matplotlib 축에 이미지 넣기
    png = drawer.GetDrawingText()
    
    # 바이너리 이미지를 읽어서 표시
    image_stream = io.BytesIO(png)
    img_array = plt.imread(image_stream)
    
    ax.imshow(img_array)
    ax.axis('off')
    
    # 제목에 Max값 표시 (검증용)
    ax.set_title(f"{title}\n(Max: {w_max:.3f})", fontsize=10, fontweight='bold')



# ==========================================
# 2. 메인 실행 로직 (Main)
# ==========================================
def main():
    # 1. 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. 데이터 파일 확인
    if not os.path.exists(DATA_FILE_PATH):
        print(f"Error: Data file not found at {DATA_FILE_PATH}")
        return

    # ---------------------------------------------------------
    # [핵심 수정] 모델 초기화 및 로드 (이 부분이 없어서 에러가 났음)
    # ---------------------------------------------------------
    print("Loading model...")
    model = LOHCGNN(
        node_in_dim=TOTAL_FEATURE_DIMENSION,
        edge_in_dim=NUM_BOND_FEATURES,
        line_edge_in_dim=NUM_LINE_EDGE_FEATURES,
        hidden_dim=HYPERPARAMS['hidden_dim'],
        num_layers=HYPERPARAMS['num_layers'],
        num_output_features=HYPERPARAMS['num_output_features'],
        dropout_rate=HYPERPARAMS['dropout_rate']
    ).to(device)

    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print("Model weights loaded successfully.")
    else:
        print(f"Warning: Model file not found at {MODEL_SAVE_PATH}. Using random weights.")

    model.eval()
    # ---------------------------------------------------------

    # 3. 데이터 로드 및 샘플링
    print("Loading data...")
    df = pd.read_excel(DATA_FILE_PATH)
    
    if len(df) < 4:
        print("Not enough data samples.")
        return

    # 무작위 4개 샘플 선택
    sample_indices = random.sample(range(len(df)), 4)
    print(f"Selected Sample Indices: {sample_indices}")

    # 4. 시각화 루프
    fig, axes = plt.subplots(4, 2, figsize=(10, 16)) # 4행 2열

    for i, idx in enumerate(sample_indices):
        row = df.iloc[idx]
        smiles_de = row[DEHYDRO_SMILES_COL]
        smiles_hy = row[HYDRO_SMILES_COL]

        if not (isinstance(smiles_de, str) and isinstance(smiles_hy, str)):
            continue

        # 여기서 'model' 변수가 정의되어 있어야 실행됨
        w_hy, w_de = get_attention_weights(model, smiles_de, smiles_hy, device)

        if w_hy is not None:
            print(f"Index {idx} - Hydro Weights: {np.round(w_hy, 4)}") # 소수점 4자리까지 확인
            print(f"Max: {w_hy.max():.4f}, Min: {w_hy.min():.4f}, Std: {w_hy.std():.4f}")
            # 왼쪽: 탈수소화 (Dehydro)
            draw_molecule_with_attention(smiles_de, w_de, axes[i][0], f"Idx {idx} [De-hydro]")
            # 오른쪽: 수소화 (Hydro)
            draw_molecule_with_attention(smiles_hy, w_hy, axes[i][1], f"Idx {idx} [Hydro]")

    plt.tight_layout()
    plt.savefig('attention_visualization_grid.png', dpi=300)
    print("Visualization saved to 'attention_visualization_grid.png'")
    plt.show()

if __name__ == "__main__":
    main()