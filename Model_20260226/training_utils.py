import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import os
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
from matplotlib.ticker import FormatStrFormatter


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    grad_clip_norm: Optional[float] = None,
    log_grad_norm: bool = False,
):
    model.train()
    total_loss = 0.0
    total_graphs = 0

    # Gradient norm stats (per batch)
    grad_norm_sum = 0.0
    grad_norm_max = 0.0
    grad_norm_count = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        if not (isinstance(batch, (list, tuple)) and len(batch) == 2):
            logging.warning("Invalid batch in training, skipping.")
            continue

        atom_batch, line_batch = batch
        atom_batch = atom_batch.to(device)
        line_batch = line_batch.to(device)

        optimizer.zero_grad()
        output, _ = model(atom_batch, line_batch)
        target = atom_batch.y.view_as(output)

        loss = criterion(output, target)
        loss.backward()

        # Gradient norm (before clipping). clip_grad_norm_ returns the total norm.
        batch_grad_norm = None
        if grad_clip_norm is not None and float(grad_clip_norm) > 0:
            batch_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm)))
        elif log_grad_norm:
            # Compute total grad norm without clipping
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is None:
                    continue
                param_norm = float(p.grad.data.norm(2))
                total_norm_sq += param_norm * param_norm
            batch_grad_norm = float(total_norm_sq ** 0.5)

        if batch_grad_norm is not None:
            grad_norm_sum += batch_grad_norm
            grad_norm_max = max(grad_norm_max, batch_grad_norm)
            grad_norm_count += 1

        optimizer.step()

        # Weight by number of graphs to make the epoch loss comparable across different last-batch sizes.
        num_g = int(getattr(atom_batch, "num_graphs", 0))
        total_loss += float(loss.item()) * max(num_g, 1)
        total_graphs += max(num_g, 1)

    avg_loss = total_loss / total_graphs if total_graphs > 0 else 0.0
    out = {
        'loss': avg_loss,
    }
    if log_grad_norm or (grad_clip_norm is not None and float(grad_clip_norm) > 0):
        out['grad_norm_mean'] = (grad_norm_sum / grad_norm_count) if grad_norm_count > 0 else float('nan')
        out['grad_norm_max'] = grad_norm_max if grad_norm_count > 0 else float('nan')
    return out


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    predictions_list, targets_list, identifiers_list = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            if not (isinstance(batch, (list, tuple)) and len(batch) == 2):
                logging.warning("Invalid batch in evaluation, skipping.")
                continue

            atom_batch, line_batch = batch
            atom_batch = atom_batch.to(device)
            line_batch = line_batch.to(device)

            num_in_batch = int(getattr(atom_batch, "num_graphs", 0))
            if hasattr(atom_batch, "mol_id") and atom_batch.mol_id is not None:
                mol_ids = atom_batch.mol_id.detach().cpu().view(-1).numpy().tolist()
                identifiers = [str(int(x)) for x in mol_ids]
            else:
                identifiers = [f"Item_{i}" for i in range(num_in_batch)]

            output, _ = model(atom_batch, line_batch)
            target = atom_batch.y.view_as(output)

            loss = criterion(output, target)

            # Weight by number of graphs for correct dataset-level averaging.
            total_loss += float(loss.item()) * max(num_in_batch, 1)
            total_graphs += max(num_in_batch, 1)

            predictions_list.append(output.cpu().numpy())
            targets_list.append(target.cpu().numpy())
            identifiers_list.extend(identifiers)

    avg_loss = total_loss / total_graphs if total_graphs > 0 else 0.0
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
        fig_loss, ax_loss = plt.subplots(figsize=(12, 8))
        ax_loss.plot(train_losses, label='Train Loss', color='royalblue', linewidth=2.5)
        ax_loss.plot(val_losses, label='Validation Loss', color='darkorange', linewidth=2.5)
        ax_loss.set_xlabel('Epoch', fontsize=28, fontweight='bold')
        ax_loss.set_ylabel('Loss (Scaled MSE)', fontsize=28, fontweight='bold')
        ax_loss.set_title(f'Training & Validation Loss Curve', fontsize=32)
        ax_loss.legend(fontsize=24)
        ax_loss.tick_params(axis='both', which='major', labelsize=24)
        ax_loss.grid(True, linestyle='--', alpha=0.6)
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
        
        fig = plt.figure(figsize=(12, 13), constrained_layout=True)
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 4])
        
        ax_histx = fig.add_subplot(gs[0, 0])
        ax_scatter = fig.add_subplot(gs[1, 0], sharex=ax_histx)

        hb = ax_scatter.hexbin(actual_valid, pred_valid, gridsize=50, cmap='inferno', norm=LogNorm())
        
        cbar = fig.colorbar(hb, ax=ax_scatter, shrink=0.8, aspect=30, pad=0.02)
        cbar.set_label('Density', fontsize=30, labelpad=0)
        cbar.ax.tick_params(labelsize=24)
        
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

        ax_histx.hist(actual_valid, bins=50, color='coral', alpha=0.7)
        plt.setp(ax_histx.get_xticklabels(), visible=False)
        ax_histx.get_yaxis().set_visible(False)

        ax_scatter.set_xlabel(f'DFT {prop_name}', fontsize=36, fontweight='bold', labelpad=20)
        ax_scatter.set_ylabel(f'LOHCGNN Prediction', fontsize=36, fontweight='bold', labelpad=20)
        ax_scatter.tick_params(axis='both', which='major', labelsize=30)

        if 'energy' in prop_name.lower():
            ax_scatter.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

        min_val = min(np.min(actual_valid), np.min(pred_valid))
        max_val = max(np.max(actual_valid), np.max(pred_valid))
        padding = (max_val - min_val) * 0.05
        plot_min = min_val - padding
        plot_max = max_val + padding
        
        ax_scatter.plot([plot_min, plot_max], [plot_min, plot_max], 'w--', lw=2.0)
        ax_scatter.set_xlim(plot_min, plot_max)
        ax_scatter.set_ylim(plot_min, plot_max)

        prop_metrics = metrics.get(prop_name, {})
        mae = prop_metrics.get('MAE', np.nan)
        r2 = prop_metrics.get('R2', np.nan)
        rmse = prop_metrics.get('RMSE', np.nan)
        
        text_str = f'R² = {r2:.2f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}'
        
        ax_scatter.text(0.05, 0.95, text_str, transform=ax_scatter.transAxes,
                        fontsize=30, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.9,
                                  edgecolor='black', linewidth=1.5))

        scatter_path = os.path.join(output_dir, f'{prop_name}_density_scatter_{base_filename}.png')
        fig.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved density scatter plot for {prop_name} to {scatter_path}")
def compute_error_percentiles(abs_err: np.ndarray, percentiles: List[int]) -> Dict[str, Any]:
    """Compute percentile stats for absolute errors.

    Args:
        abs_err: shape (N, T) or (N,)
        percentiles: e.g., [50, 90, 95, 99]

    Returns:
        dict with keys: pXX for each percentile, plus mean and max
    """
    if abs_err.size == 0:
        return {"mean": np.nan, "max": np.nan, **{f"p{p}": np.nan for p in percentiles}}

    flat = abs_err.reshape(-1)
    out: Dict[str, Any] = {"mean": float(np.mean(flat)), "max": float(np.max(flat))}
    for p in percentiles:
        out[f"p{int(p)}"] = float(np.percentile(flat, p))
    return out


def get_worst_k_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    identifiers: List[str],
    k: int = 10,
    reduce: str = "max",
) -> List[Dict[str, Any]]:
    """Return worst-k samples by absolute error.

    Args:
        y_true, y_pred: shape (N, T)
        identifiers: length N
        reduce: {"max", "mean"} to aggregate per-target errors into a per-sample score

    Returns:
        list of dict rows (sorted worst->best among the k).
    """
    if y_true.size == 0 or y_pred.size == 0 or y_true.shape != y_pred.shape:
        return []
    if len(identifiers) != y_true.shape[0]:
        return []

    err = (y_pred - y_true)
    abs_err = np.abs(err)
    if reduce == "mean":
        score = abs_err.mean(axis=1)
    else:
        score = abs_err.max(axis=1)

    idx = np.argsort(-score)[: max(int(k), 1)]
    rows: List[Dict[str, Any]] = []
    for i in idx:
        rows.append({
            "id": identifiers[i],
            "score": float(score[i]),
            "true": y_true[i].tolist(),
            "pred": y_pred[i].tolist(),
            "err": err[i].tolist(),
            "abs_err": abs_err[i].tolist(),
        })
    return rows


def save_worst_k_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        return
    import csv
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    keys = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
