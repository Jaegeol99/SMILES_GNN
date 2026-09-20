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
    
    for atom_batch, line_batch in tqdm(loader, desc="Training", leave=False):
        atom_batch = atom_batch.to(device)
        line_batch = line_batch.to(device)

        optimizer.zero_grad()
        output = model(atom_batch, line_batch)
        target = atom_batch.y.view_as(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader) if len(loader) > 0 else 0.0

def evaluate_epoch(model: nn.Module,
                   loader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device) -> Tuple[float, np.ndarray, np.ndarray, List[Any]]:
    model.eval()
    total_loss = 0.0
    predictions_list, targets_list, identifiers_list = [], [], []

    with torch.no_grad():
        for atom_batch, line_batch in tqdm(loader, desc="Evaluating", leave=False):
            atom_batch = atom_batch.to(device)
            line_batch = line_batch.to(device)
            
            num_in_batch = getattr(atom_batch, 'num_graphs', 0)
            identifiers = [f"Item_{i}" for i in range(num_in_batch)]

            output = model(atom_batch, line_batch)
            target = atom_batch.y.view_as(output)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            predictions_list.append(output.cpu().numpy())
            targets_list.append(target.cpu().numpy())
            identifiers_list.extend(identifiers)

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
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
            results[prop_name] = {'MSE': np.nan, 'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan}
            continue

        mse = mean_squared_error(actual_valid, pred_valid)
        r2 = r2_score(actual_valid, pred_valid)
        mae = mean_absolute_error(actual_valid, pred_valid)
        rmse = np.sqrt(mse)
        results[prop_name] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

        logging.info(f"Metrics for {prop_name}:")
        logging.info(f"  R2: {r2:.4f}")
        logging.info(f"  MAE: {mae:.4f}")
        logging.info(f"  RMSE: {rmse:.4f}")
        logging.info(f"  MSE: {mse:.4f}")

    return results


def plot_results(train_losses: List[float], val_losses: List[float],
                 actual_original: np.ndarray, pred_original: np.ndarray,
                 property_names: List[str],
                 metrics: Dict[str, Dict[str, float]],
                 output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    base_filename = "model"

    if train_losses and val_losses:
        fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
        ax_loss.plot(train_losses, label='Train Loss', color='royalblue', linewidth=2)
        ax_loss.plot(val_losses, label='Validation Loss', color='darkorange', linewidth=2)
        ax_loss.set_xlabel('Epoch', fontsize=12)
        ax_loss.set_ylabel('Loss (Scaled MSE)', fontsize=12)
        ax_loss.set_title(f'Training & Validation Loss Curve ({base_filename})', fontsize=14)
        ax_loss.legend(fontsize=10)
        ax_loss.grid(True)
        ax_loss.set_ylim(bottom=0)
        
        loss_curve_path = os.path.join(output_dir, f'loss_curve_{base_filename}.png')
        fig_loss.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
        plt.close(fig_loss)
        logging.info(f"Saved loss curve to {loss_curve_path}")
    else:
        logging.info("Loss history not provided, skipping loss curve plot.")

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
        
        fig = plt.figure(figsize=(8, 9), constrained_layout=True)
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 4])
        
        ax_histx = fig.add_subplot(gs[0, 0])
        ax_scatter = fig.add_subplot(gs[1, 0], sharex=ax_histx)

        hb = ax_scatter.hexbin(actual_valid, pred_valid, gridsize=50, cmap='inferno', norm=LogNorm())
        cbar = fig.colorbar(hb, ax=ax_scatter, shrink=0.8, aspect=30, pad=0.02)
        cbar.set_label('Density', size=12)

        ax_histx.hist(actual_valid, bins=50, color='coral', alpha=0.7)
        plt.setp(ax_histx.get_xticklabels(), visible=False)
        ax_histx.get_yaxis().set_visible(False)

        ax_scatter.set_xlabel(f'Experimental {prop_name}', fontsize=16)
        ax_scatter.set_ylabel(f'LOHCGNN {prop_name}', fontsize=16)

        min_val = min(np.min(actual_valid), np.min(pred_valid))
        max_val = max(np.max(actual_valid), np.max(pred_valid))
        padding = (max_val - min_val) * 0.05
        plot_min = min_val - padding
        plot_max = max_val + padding
        
        ax_scatter.plot([plot_min, plot_max], [plot_min, plot_max], 'w--', lw=1.5)
        ax_scatter.set_xlim(plot_min, plot_max)
        ax_scatter.set_ylim(plot_min, plot_max)

        prop_metrics = metrics.get(prop_name, {})
        mae = prop_metrics.get('MAE', np.nan)
        r2 = prop_metrics.get('R2', np.nan)
        rmse = prop_metrics.get('RMSE', np.nan)
        text_str = f'R² = {r2:.2f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}'
        
        ax_scatter.text(0.05, 0.95, text_str, transform=ax_scatter.transAxes,
                        fontsize=14, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.9,
                                  edgecolor='black', linewidth=1.5))

        scatter_path = os.path.join(output_dir, f'{prop_name}_density_scatter_{base_filename}.png')
        fig.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved density scatter plot for {prop_name} to {scatter_path}")