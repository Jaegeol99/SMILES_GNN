import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt
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
                 property_names: List[str], output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    base_filename = "model"

    # Plot loss curves
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

        fig_scatter, ax_scatter = plt.subplots(figsize=(6, 6))
        ax_scatter.scatter(actual_valid, pred_valid, alpha=0.7, s=30, color='deepskyblue')
        ax_scatter.set_xlabel(f'Actual {prop_name}', fontsize=14)
        ax_scatter.set_ylabel(f'Predicted {prop_name}', fontsize=14)
        ax_scatter.set_title(f'{prop_name}: Prediction vs Actual ({base_filename})', fontsize=16)
        
        min_val = min(np.min(actual_valid), np.min(pred_valid))
        max_val = max(np.max(actual_valid), np.max(pred_valid))
        padding = (max_val - min_val) * 0.05
        plot_min = min_val - padding
        plot_max = max_val + padding
        ax_scatter.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=1.5, label='y=x')
        ax_scatter.set_xlim(plot_min, plot_max)
        ax_scatter.set_ylim(plot_min, plot_max)
        ax_scatter.legend(fontsize=10)
        
        fig_scatter.tight_layout()
        scatter_path = os.path.join(output_dir, f'{prop_name}_scatter_{base_filename}.png')
        fig_scatter.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close(fig_scatter)
        logging.info(f"Saved scatter plot for {prop_name} to {scatter_path}")