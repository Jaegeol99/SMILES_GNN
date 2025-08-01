# training_utils.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import os
from typing import List, Tuple, Dict
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        if not hasattr(batch, 'y') or batch.y is None: continue
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.y.view_as(output))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader) if loader else 0.0

def evaluate_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    predictions, targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            if not hasattr(batch, 'y') or batch.y is None: continue
            output = model(batch)
            loss = criterion(output, batch.y.view_as(output))
            total_loss += loss.item()
            predictions.append(output.cpu().numpy())
            targets.append(batch.y.cpu().numpy())
    avg_loss = total_loss / len(loader) if loader else 0.0
    return avg_loss, np.concatenate(predictions), np.concatenate(targets)

def evaluate_metrics(predictions: np.ndarray, actuals: np.ndarray, prop_name: str) -> Dict[str, float]:
    if actuals.size < 2: return {'MSE': np.nan, 'R2': np.nan, 'MAE': np.nan}
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    logging.info(f"Metrics for {prop_name}: MSE={mse:.4f}, R2={r2:.4f}, MAE={mae:.4f}")
    return {'MSE': mse, 'R2': r2, 'MAE': mae}

def plot_results(train_losses: List[float], test_losses: List[float], actuals: np.ndarray, preds: np.ndarray, prop_name: str, metrics: Dict[str, float], output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    base_filename = "atomic_charge_model"
    
    # Loss Curve Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training & Test Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'loss_curve_{base_filename}.png'), dpi=300)
    plt.close()

    # Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.hexbin(actuals, preds, gridsize=50, cmap='inferno', norm=LogNorm())
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'w--')
    plt.xlabel(f'DFT {prop_name}')
    plt.ylabel(f'GNN {prop_name}')
    plt.title(f'{prop_name} Prediction: Actual vs. Predicted')
    plt.colorbar(label='Density')
    text_str = f"MAE = {metrics['MAE']:.4f}\n$R^2$ = {metrics['R2']:.3f}"
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.savefig(os.path.join(output_dir, f'scatter_{prop_name}_{base_filename}.png'), dpi=300)
    plt.close()