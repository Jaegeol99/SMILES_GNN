import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import os
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

def train_epoch(model: nn.Module,
                loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    batch_count = 0

    for data_pair in tqdm(loader, desc="Training", leave=False):
        if not (isinstance(data_pair, (list, tuple)) and len(data_pair) == 2):
            logging.warning("Invalid data pair in training batch, skipping.")
            continue
        
        (atom_de, line_de), (atom_hy, line_hy) = data_pair
        
        atom_batch = atom_hy.to(device)
        atom_batch.x_de = atom_de.x.to(device)
        atom_batch.edge_index_de = atom_de.edge_index.to(device)
        atom_batch.edge_attr_de = atom_de.edge_attr.to(device)
        atom_batch.batch_de = atom_de.batch.to(device)

        line_batch = line_hy.to(device)
        line_batch.x_de = line_de.x.to(device)
        line_batch.edge_index_de = line_de.edge_index.to(device)
        line_batch.edge_attr_de = line_de.edge_attr.to(device)
        line_batch.batch_de = line_de.batch.to(device)

        optimizer.zero_grad()
        output = model(atom_batch, line_batch)
        target = atom_batch.y.view_as(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    return total_loss / batch_count if batch_count > 0 else 0.0

def evaluate_epoch(model: nn.Module,
                   loader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device) -> Tuple[float, np.ndarray, np.ndarray, List[Any]]:
    model.eval()
    total_loss = 0.0
    predictions_list, targets_list, identifiers_list = [], [], []
    batch_count = 0

    with torch.no_grad():
        for data_pair in tqdm(loader, desc="Evaluating", leave=False):
            if not (isinstance(data_pair, (list, tuple)) and len(data_pair) == 2):
                logging.warning("Invalid data pair in evaluation batch, skipping.")
                continue
            
            (atom_de, line_de), (atom_hy, line_hy) = data_pair

            atom_batch = atom_hy.to(device)
            atom_batch.x_de = atom_de.x.to(device)
            atom_batch.edge_index_de = atom_de.edge_index.to(device)
            atom_batch.edge_attr_de = atom_de.edge_attr.to(device)
            atom_batch.batch_de = atom_de.batch.to(device)

            line_batch = line_hy.to(device)
            line_batch.x_de = line_de.x.to(device)
            line_batch.edge_index_de = line_de.edge_index.to(device)
            line_batch.edge_attr_de = line_de.edge_attr.to(device)
            line_batch.batch_de = line_de.batch.to(device)
            
            num_in_batch = getattr(atom_batch, 'num_graphs', 0)
            identifiers = [f"Item_{i}" for i in range(num_in_batch)]

            output = model(atom_batch, line_batch)
            target = atom_batch.y.view_as(output)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            predictions_list.append(output.cpu().numpy())
            targets_list.append(target.cpu().numpy())
            identifiers_list.extend(identifiers)
            batch_count += 1

    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    predictions = np.concatenate(predictions_list, axis=0) if predictions_list else np.array([])
    targets = np.concatenate(targets_list, axis=0) if targets_list else np.array([])

    return avg_loss, predictions, targets, identifiers_list

def evaluate_metrics(predictions: np.ndarray,
                     actuals: np.ndarray,
                     property_names: List[str]) -> Dict[str, Dict[str, float]]:
    results = {}
    if predictions.size == 0 or actuals.size == 0 or predictions.shape != actuals.shape:
        logging.warning("No valid predictions or actual values to evaluate.")
        return results

    num_properties = predictions.shape[1]
    property_names_used = property_names if len(property_names) == num_properties \
                                      else [f"Property_{i}" for i in range(num_properties)]

    for i in range(num_properties):
        prop_name = property_names_used[i]
        actual_column = actuals[:, i]
        pred_column = predictions[:, i]
        valid_mask = ~np.isnan(actual_column) & ~np.isnan(pred_column)
        actual_valid = actual_column[valid_mask]
        pred_valid = pred_column[valid_mask]

        if actual_valid.size < 2:
            logging.warning(f"Insufficient valid data for property: {prop_name}")
            results[prop_name] = {'MSE': np.nan, 'R2': np.nan, 'MAE': np.nan}
            continue

        mse = mean_squared_error(actual_valid, pred_valid)
        r2 = r2_score(actual_valid, pred_valid)
        mae = mean_absolute_error(actual_valid, pred_valid)
        results[prop_name] = {'MSE': mse, 'R2': r2, 'MAE': mae}

        logging.info(f"Metrics for {prop_name}:")
        logging.info(f"  MSE: {mse:.4f}")
        logging.info(f"  R2: {r2:.4f}")
        logging.info(f"  MAE: {mae:.4f}")

    return results

def plot_results(train_losses: List[float], test_losses: List[float],
                 actual_original: np.ndarray, pred_original: np.ndarray,
                 property_names: List[str],
                 metrics: Dict[str, Dict[str, float]],
                 output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    base_filename = "model"

    # 손실 곡선 플롯 (변경 없음)
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    ax_loss.plot(train_losses, label='Train Loss', color='royalblue', linewidth=2)
    ax_loss.plot(test_losses, label='Test Loss', color='darkorange', linewidth=2)
    ax_loss.set_xlabel('Epoch', fontsize=12)
    ax_loss.set_ylabel('Loss (Scaled MSE)', fontsize=12)
    ax_loss.set_title(f'Training & Test Loss Curve ({base_filename})', fontsize=14)
    ax_loss.legend(fontsize=10)
    ax_loss.grid(True)
    ax_loss.set_ylim(bottom=0)
    
    loss_curve_path = os.path.join(output_dir, f'loss_curve_{base_filename}.png')
    fig_loss.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close(fig_loss)
    logging.info(f"Saved loss curve to {loss_curve_path}")

    if pred_original.size == 0 or actual_original.size == 0 or pred_original.shape != actual_original.shape:
        logging.warning("No valid data for plotting predictions vs actuals.")
        return

    num_properties = pred_original.shape[1]
    property_names_plot = property_names if len(property_names) == num_properties \
                                        else [f"Target_{i+1}" for i in range(num_properties)]

    for i in range(num_properties):
        prop_name = property_names_plot[i]
        actual = actual_original[:, i]
        pred = pred_original[:, i]

        valid_mask = ~np.isnan(actual) & ~np.isnan(pred)
        actual_valid = actual[valid_mask]
        pred_valid = pred[valid_mask]

        if actual_valid.size == 0:
            logging.warning(f"No valid data to plot for {prop_name}")
            continue

        # ---▼▼▼ 플롯 생성 로직 수정 ▼▼▼---
        
        # 1. 레이아웃 설정: 2행 1열 구조로 변경
        fig = plt.figure(figsize=(8, 9), constrained_layout=True)
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 4])
        
        ax_histx = fig.add_subplot(gs[0, 0])
        ax_scatter = fig.add_subplot(gs[1, 0], sharex=ax_histx)

        # 2. 메인 hexbin 플롯
        # cmap='inferno'는 제공된 이미지와 유사한 색상 맵입니다.
        hb = ax_scatter.hexbin(actual_valid, pred_valid, gridsize=50, cmap='inferno', norm=LogNorm())

        # 3. 색상 막대 (더 작게, 오른쪽에)
        # shrink와 aspect를 조절하여 크기와 비율을 맞춥니다.
        cbar = fig.colorbar(hb, ax=ax_scatter, shrink=0.8, aspect=30, pad=0.02)
        cbar.set_label('Density', size=12)

        # 4. 히스토그램
        ax_histx.hist(actual_valid, bins=50, color='coral', alpha=0.7)
        plt.setp(ax_histx.get_xticklabels(), visible=False) # 히스토그램의 x축 눈금 숨기기
        ax_histx.get_yaxis().set_visible(False) # 히스토그램의 y축 숨기기

        # 5. 레이블 및 제목 변경/제거
        # 제목 제거 (ax_histx.set_title(...) 호출 안 함)
        ax_scatter.set_xlabel(f'DFT {prop_name}', fontsize=16) # X축 레이블 변경
        ax_scatter.set_ylabel(f'LOHCGNN {prop_name}', fontsize=16) # Y축 레이블 변경

        # 6. y=x 선 및 범위 설정
        min_val = min(np.min(actual_valid), np.min(pred_valid))
        max_val = max(np.max(actual_valid), np.max(pred_valid))
        padding = (max_val - min_val) * 0.05
        plot_min = min_val - padding
        plot_max = max_val + padding
        
        # y=x 선 (범례 없이)
        ax_scatter.plot([plot_min, plot_max], [plot_min, plot_max], 'w--', lw=1.5)
        ax_scatter.set_xlim(plot_min, plot_max)
        ax_scatter.set_ylim(plot_min, plot_max)
        # 범례 제거 (ax_scatter.legend(...) 호출 안 함)

        # 7. MAE와 R2 텍스트 추가 (변경 없음)
        prop_metrics = metrics.get(prop_name, {})
        mae = prop_metrics.get('MAE', np.nan)
        r2 = prop_metrics.get('R2', np.nan)
        text_str = f'MAE = {mae:.4f}\n$R^2$ = {r2:.3f}'
        ax_scatter.text(0.05, 0.95, text_str, transform=ax_scatter.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8))

        # 8. 이미지 저장
        scatter_path = os.path.join(output_dir, f'{prop_name}_density_scatter_{base_filename}.png')
        fig.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved density scatter plot for {prop_name} to {scatter_path}")
        # ---▲▲▲ 플롯 생성 로직 수정 종료 ▲▲▲---