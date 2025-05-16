# train.py
"""
Train and evaluate the LOHC GNN multi-target energy prediction model.
Includes data loading, preprocessing, training loop, evaluation, plotting, and saving.
"""

import os
import logging
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Project modules and configuration
from config import (
    DATA_FILE_PATH, DEHYDRO_SMILES_COL, HYDRO_SMILES_COL, LABEL_COLS,
    HYPERPARAMS, PROPERTY_NAMES, MODEL_SAVE_PATH, LABEL_SCALING_PARAMS_PATH,
    TOTAL_FEATURE_DIMENSION, LOGGING_LEVEL, LOG_FORMAT
)
from data_preprocessing import (
    smiles_to_graph_data, initialize_feature_scaling
)
from gnn_model import PairedLOHCGNN

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL.upper(), logging.INFO),
    format=LOG_FORMAT
)

PairedData = Tuple[Any, Any]


def load_and_preprocess_paired_data(
    path: str, de_col: str, hy_col: str, label_cols: List[str]
) -> Tuple[List[PairedData], int]:
    """
    Load Excel data, create paired graph Data objects, initialize feature scaling.
    Returns list of (graph_de, graph_hy) and feature_dimension.
    """
    if not os.path.exists(path):
        logging.error(f"Data file not found: {path}")
        return [], -1

    df = pd.read_excel(path)
    missing = [c for c in [de_col, hy_col] + label_cols if c not in df.columns]
    if missing:
        logging.error(f"Missing columns: {missing}")
        return [], -1

    # Initialize scaling
    feature_dim = TOTAL_FEATURE_DIMENSION
    initialize_feature_scaling(feature_dim)

    paired_list: List[PairedData] = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing pairs"):
        de_smiles, hy_smiles = row[de_col], row[hy_col]
        if not isinstance(de_smiles, str) or not isinstance(hy_smiles, str):
            continue
        try:
            labels = [float(v) for v in row[label_cols].tolist()]
        except Exception:
            continue

        de_graph = smiles_to_graph_data(de_smiles, labels)
        hy_graph = smiles_to_graph_data(hy_smiles, labels)
        if de_graph and hy_graph:
            for g, sm in [(de_graph, de_smiles), (hy_graph, hy_smiles)]:
                g.original_index = idx
                g.smiles = sm
            paired_list.append((de_graph, hy_graph))

    logging.info(f"Processed {len(paired_list)} valid pairs.")
    return paired_list, feature_dim


def calculate_label_scaling_params(
    data: List[PairedData], train_idx: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute minimum and maximum label values across training data for scaling.
    """
    labels = []
    for i in train_idx:
        y = data[i][0].y.numpy()
        if not np.isnan(y).any():
            labels.append(y)
    all_labels = np.vstack(labels)
    min_vals = np.nanmin(all_labels, axis=0)
    max_vals = np.nanmax(all_labels, axis=0)
    # avoid zero range
    range_ = max_vals - min_vals
    range_[np.isclose(range_, 0)] = 1.0
    return min_vals, max_vals


def apply_label_scaling(
    data: List[PairedData], min_vals: np.ndarray, max_vals: np.ndarray
):
    """
    In-place min-max scale labels for all paired graphs.
    """
    range_ = max_vals - min_vals
    for de_graph, hy_graph in data:
        y = de_graph.y.numpy()
        scaled = (y - min_vals) / range_
        tensor = torch.tensor(scaled, dtype=torch.float)
        de_graph.y = tensor
        hy_graph.y = tensor.clone()


def create_dataloaders(
    data: List[PairedData], train_idx: List[int], test_idx: List[int], batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """
    Build DataLoader for training and testing from paired graph list.
    """
    train_pairs = [data[i] for i in train_idx]
    test_pairs = [data[i] for i in test_idx]
    # flatten paired graphs
    train_graphs = sum(([d, h] for d, h in train_pairs), [])
    test_graphs = sum(([d, h] for d, h in test_pairs), [])

    return (
        DataLoader(train_graphs, batch_size=batch_size, shuffle=True),
        DataLoader(test_graphs, batch_size=batch_size)
    )


def train_epoch(model: nn.Module, loader: DataLoader,
                criterion, optimizer) -> float:
    """
    Single training epoch over DataLoader.
    """
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(model.device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def evaluate_epoch(model: nn.Module, loader: DataLoader,
                   criterion) -> float:
    """
    Single evaluation epoch (no grad).
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(model.device)
            pred = model(batch)
            loss = criterion(pred, batch.y)
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def plot_results(
    train_losses: List[float], test_losses: List[float],
    actual: np.ndarray, preds: np.ndarray, names: List[str]
):
    """
    Plot loss curves and prediction vs actual scatter.
    """
    os.makedirs("plots", exist_ok=True)
    # Loss curve
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig('plots/loss_curve.png')
    plt.close()

    # Scatter per property
    for i, name in enumerate(names):
        plt.figure()
        plt.scatter(actual[:, i], preds[:, i], alpha=0.6)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{name}: Actual vs Predicted')
        plt.savefig(f'plots/{name}_scatter.png')
        plt.close()


def main():
    """Execute full training pipeline."""
    logging.info(f"Starting training: {MODEL_SAVE_PATH}")

    data_pairs, feat_dim = load_and_preprocess_paired_data(
        DATA_FILE_PATH, DEHYDRO_SMILES_COL, HYDRO_SMILES_COL, LABEL_COLS
    )
    if not data_pairs or feat_dim <= 0:
        return

    # Split indices
    indices = list(range(len(data_pairs)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=HYPERPARAMS['test_split_ratio'],
        random_state=HYPERPARAMS['random_state']
    )

    # Label scaling
    min_vals, max_vals = calculate_label_scaling_params(data_pairs, train_idx)
    apply_label_scaling(data_pairs, min_vals, max_vals)
    np.savez(LABEL_SCALING_PARAMS_PATH, min=min_vals, max=max_vals)

    # DataLoaders
    train_loader, test_loader = create_dataloaders(
        data_pairs, train_idx, test_idx, HYPERPARAMS['batch_size']
    )

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PairedLOHCGNN(
        num_node_features=feat_dim,
        hidden_channels=HYPERPARAMS['hidden_channels'],
        num_output_features=HYPERPARAMS['num_output_features'],
        dropout_rate=HYPERPARAMS['dropout_rate'],
        gnn_layers=HYPERPARAMS['gnn_layers'],
        gat_heads=HYPERPARAMS['gat_heads'],
        gat_output_heads=HYPERPARAMS['gat_output_heads']
    ).to(device)
    model.device = device
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS['learning_rate'])

    # Training loop
    train_losses, test_losses = [], []
    for epoch in range(1, HYPERPARAMS['epochs'] + 1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer)
        te_loss = evaluate_epoch(model, test_loader, criterion)
        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}: Train={tr_loss:.4f}, Test={te_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logging.info(f"Model saved to {MODEL_SAVE_PATH}")

    # Final evaluation and plotting
    # Gather all test predictions
    actual, preds = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            actual.append(batch.y.cpu().numpy())
            preds.append(out.cpu().numpy())
    actual = np.vstack(actual)
    preds = np.vstack(preds)

    # Metrics
    for i, name in enumerate(PROPERTY_NAMES):
        mse = mean_squared_error(actual[:, i], preds[:, i])
        r2 = r2_score(actual[:, i], preds[:, i])
        logging.info(f"{name}: MSE={mse:.4f}, R2={r2:.4f}")

    plot_results(train_losses, test_losses, actual, preds, PROPERTY_NAMES)


if __name__ == '__main__':
    main()
