# training_utils.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os, logging

def prepare_batch(data_pair, device):
    (atom_de, line_de), (atom_hy, line_hy) = data_pair
    ah = atom_hy.to(device)
    ah.x_de, ah.edge_index_de, ah.edge_attr_de, ah.batch_de = atom_de.x.to(device), atom_de.edge_index.to(device), atom_de.edge_attr.to(device), atom_de.batch.to(device)
    lh = line_hy.to(device)
    lh.x_de, lh.edge_index_de, lh.edge_attr_de, lh.batch_de = line_de.x.to(device), line_de.edge_index.to(device), line_de.edge_attr.to(device), line_de.batch.to(device)
    return ah, lh

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        ah, lh = prepare_batch(batch, device)
        optimizer.zero_grad()
        out = model(ah, lh)
        loss = criterion(out, ah.y.view_as(out))
        loss.backward(); optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, preds, targs = 0, [], []
    with torch.no_grad():
        for batch in loader:
            ah, lh = prepare_batch(batch, device)
            out = model(ah, lh)
            total_loss += criterion(out, ah.y.view_as(out)).item()
            preds.append(out.cpu().numpy()); targs.append(ah.y.view_as(out).cpu().numpy())
    return total_loss / len(loader), np.concatenate(preds), np.concatenate(targs), []

def evaluate_metrics(p, a, names):
    res = {}
    for i, n in enumerate(names):
        pi, ai = p[:, i], a[:, i]
        mask = ~np.isnan(ai)
        pi, ai = pi[mask], ai[mask]
        mse = mean_squared_error(ai, pi)
        res[n] = {'MSE': mse, 'R2': r2_score(ai, pi), 'MAE': mean_absolute_error(ai, pi), 'RMSE': np.sqrt(mse)}
        logging.info(f"{n} - R2: {res[n]['R2']:.4f}, MAE: {res[n]['MAE']:.4f}")
    return res

def plot_results(tr_l, val_l, a, p, names, metrics, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Loss Curve (High Quality)
    fig_loss, ax_loss = plt.subplots(figsize=(12, 8))
    ax_loss.plot(tr_l, label='Train Loss', color='royalblue', linewidth=2.5)
    ax_loss.plot(val_l, label='Validation Loss', color='darkorange', linewidth=2.5)
    ax_loss.set_xlabel('Epoch', fontsize=28, fontweight='bold')
    ax_loss.set_ylabel('Loss (Scaled MSE)', fontsize=28, fontweight='bold')
    ax_loss.legend(fontsize=24); ax_loss.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{output_dir}/loss_curve.png", dpi=300, bbox_inches='tight'); plt.close()
    
    # 2. Scatter Plots (High Quality - Hexbin with Marginal Histograms)
    for i, n in enumerate(names):
        actual, pred = a[:, i], p[:, i]
        mask = ~np.isnan(actual) & ~np.isnan(pred)
        actual, pred = actual[mask], pred[mask]
        if len(actual) == 0: continue

        fig = plt.figure(figsize=(12, 13), constrained_layout=True)
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 4])
        ax_histx = fig.add_subplot(gs[0, 0])
        ax_scatter = fig.add_subplot(gs[1, 0], sharex=ax_histx)

        # Hexbin plot
        hb = ax_scatter.hexbin(actual, pred, gridsize=50, cmap='inferno', norm=LogNorm())
        cbar = fig.colorbar(hb, ax=ax_scatter, shrink=0.8, aspect=30, pad=0.02)
        cbar.set_label('Density', fontsize=30); cbar.ax.tick_params(labelsize=24)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

        # Marginal Histogram
        ax_histx.hist(actual, bins=50, color='coral', alpha=0.7)
        plt.setp(ax_histx.get_xticklabels(), visible=False); ax_histx.get_yaxis().set_visible(False)

        ax_scatter.set_xlabel(f'DFT {n}', fontsize=36, fontweight='bold', labelpad=20)
        ax_scatter.set_ylabel(f'Prediction', fontsize=36, fontweight='bold', labelpad=20)
        ax_scatter.tick_params(axis='both', which='major', labelsize=30)

        # y=x line
        mn, mx = min(actual.min(), pred.min()), max(actual.max(), pred.max())
        pad = (mx - mn) * 0.05
        ax_scatter.plot([mn-pad, mx+pad], [mn-pad, mx+pad], 'w--', lw=2.0)
        ax_scatter.set_xlim(mn-pad, mx+pad); ax_scatter.set_ylim(mn-pad, mx+pad)

        # Metrics Text
        m = metrics[n]
        text_str = f"R² = {m['R2']:.3f}\nMAE = {m['MAE']:.2f}\nRMSE = {m['RMSE']:.2f}"
        ax_scatter.text(0.05, 0.95, text_str, transform=ax_scatter.transAxes, fontsize=30, 
                        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.9))

        safe_name = n.replace(" ", "_")
        plt.savefig(f"{output_dir}/scatter_{safe_name}.png", dpi=300, bbox_inches='tight'); plt.close()
        logging.info(f"Saved high-quality scatter plot for {n}")