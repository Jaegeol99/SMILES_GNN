# GNN-based Gibbs Free Energy Prediction from SMILES

**A deep learning framework for predicting molecular Gibbs free energy directly from SMILES, inspired by quantum chemistry workflows such as CP2K.**

---

## Overview

이 프로젝트는 분자 구조의 **SMILES(Simplified Molecular Input Line Entry System)** 문자열을 입력받아,
Graph Neural Network(GNN)를 이용하여 Gibbs free energy(및 기타 열역학적 property)를 예측하는 머신러닝 파이프라인입니다.

이 시스템은 **양자화학 계산(CP2K 등)의 데이터셋**에서 추출된 레이블(Gibbs energy 등)을 활용하여,
CP2K처럼 다양한 분자 구조 및 화학 시스템을 빠르게 예측할 수 있는 AI surrogate 모델을 제공합니다.

---

## Key Features

* **End-to-End Prediction:**
  SMILES만 입력하면, 분자의 **Gibbs 자유 에너지** 및 추가적인 property를 즉시 예측
* **Advanced Graph Featurization:**

  * 원자 및 결합 수준의 다양한 descriptor 자동 추출
  * 이웃 원자 분포, heteroatom 거리, 함수기/모티프 서브그래프 인식
  * RDKit을 통한 SMARTS 패턴 기반 특징 추출
* **Multi-Target Learning:**
  하나의 모델에서 여러 thermodynamic/quantum property 동시 예측 가능
* **Paired Graph Modeling:**
  Hydrogenated/Dehydrogenated 페어 데이터를 위한 siamese GNN 아키텍처 제공
* **Configurable & Reproducible:**
  하이퍼파라미터, 입력 피처, 스케일링 등 모든 설정은 `config.py`에서 관리

---

## Source Code Organization

```
.
├── config.py               # 전체 설정, 하이퍼파라미터, feature 스케일 기준
├── data_preprocessing.py   # SMILES → PyG Data 변환, feature 추출
├── gnn_model.py            # GNN(특히 GATConv 기반) 모델 구조 정의
├── train.py                # 데이터 로딩, 학습/테스트, 성능 평가, 플롯
├── QM9.py                  # QM9 벤치마크 실험 (예시, optional)
└── lohc_data.xlsx          # 학습/검증용 데이터 (Excel/CSV)
```

---

## Installation

### Requirements

* Python 3.8+
* [RDKit](https://www.rdkit.org/) (분자 feature 추출용)
* [PyTorch](https://pytorch.org/)
* [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/)
* 기타: pandas, numpy, scikit-learn, matplotlib

### Setup

```bash
pip install -r requirements.txt
# 또는
conda env create -f environment.yml
```

---

## Usage

### 1. 데이터 준비

* `lohc_data.xlsx` (또는 CSV):
  필수 컬럼

  * `Dehydrogenated_SMILES`, `Hydrogenated_SMILES`
  * `Dehydrogenated_energy`, `Hydrogenated_energy`, `Potential` 등 예측할 property

### 2. 모델 학습/예측

```bash
python train.py
```

* 전체 파이프라인:

  1. 데이터셋 로드 및 전처리 (SMILES → 그래프 변환, feature scaling)
  2. train/test split 및 label scaling
  3. GNN 모델 학습
  4. 예측 및 평가, 결과 시각화

### 3. 예측 결과

* Gibbs free energy 및 추가 property의 예측 값과,
  실제 값과의 scatter plot, loss curve 등 자동 저장

---

## Model Architecture

* **입력:**

  * SMILES 기반 분자 구조
  * 원자/결합/분자 레벨 descriptor 벡터 (자동 생성)
* **특징 추출:**

  * Atomic number, degree, formal charge, hydrogen 수, aromaticity, mass
  * Heteroatom 거리 (N, O, S, B 등)
  * 이웃 원자 수, 함수기/모티프 SMARTS 패턴
* **GNN 인코더:**

  * 여러 층의 GATConv(Graph Attention) → global max pooling
  * Siamese 구조 (Hydrogenated/Dehydrogenated 페어)
* **MLP Head:**

  * Fully connected layer로 multi-target property 예측
  * Loss: MSE

**전체 플로우:**
SMILES → PyG Graph (with features) → GNN Encoder → \[Dehydro, Hydro embedding] → MLP → Energy 예측

---

## Example: 코드 흐름 요약

```python
# config.py: feature/모델 설정 및 데이터 경로 등
# data_preprocessing.py: SMILES를 PyG Data로 변환 (x, edge_index, edge_attr, y)
# gnn_model.py: GNNEncoder(GATConv+Pool), PairedLOHCGNN(페어 처리)
# train.py: 전체 학습 및 평가, 시각화
```

---

## Links

* [CP2K.org](https://www.cp2k.org/)
  과학적 reference, 원본 계산 수행 프로그램, 입력 키워드/매뉴얼 등
* [RDKit Documentation](https://www.rdkit.org/docs/)
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

---

## Directory Structure

* **arch:** (예시) 다양한 하드웨어/컴파일 설정
* **benchmarks:** 벤치마크용 입력 파일
* **data:** 기본 basis set, pseudopotential 등 (예시)
* **src:** 소스 코드
* **tools:** 데이터 전처리, 결과 분석 등 보조 스크립트
* **tests:** 회귀테스트, 단위테스트
  (실제 구현에 따라 디렉토리 구성이 상이할 수 있습니다.)

---

## Acknowledgments

* 본 프로젝트는 양자화학 계산(CP2K 등)에서 추출한 데이터셋,
  RDKit 및 PyTorch Geometric 오픈소스 라이브러리를 기반으로 개발되었습니다.

---

## 참고 사항 및 한계

* 본 모델은 **SMILES 구조로부터만 예측**이 가능하며,
  실험적으로 얻은 property(HOMO, LUMO 등)를 별도 feature로 사용할 경우
  train/test input 일치성 원칙을 반드시 지켜야 합니다.
* 예측 정확도와 일반화는 **학습 데이터 품질, 구조 다양성, feature engineering**에 크게 의존합니다.
* 논문 등 1차 문헌 참고 권장:

  * Gilmer et al., 2017, "Neural Message Passing for Quantum Chemistry"
  * Hu et al., 2020, "Strategies for Pre-training Graph Neural Networks"
