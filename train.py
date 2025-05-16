# --- START OF FILE train.py ---

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch # Explicit imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns # Keep seaborn import if used for plotting
import pandas as pd
from rdkit import Chem
import os
import logging
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm # For progress bars

# --- Import User Modules and Config ---
try:
    # Import necessary constants from config.py
    from config import (
        DATA_FILE_PATH, DEHYDRO_SMILES_COL, HYDRO_SMILES_COL, LABEL_COLS,
        HYPERPARAMS, PROPERTY_NAMES, MODEL_SAVE_PATH, LABEL_SCALING_PARAMS_PATH,
        LOGGING_LEVEL, LOG_FORMAT,
        TOTAL_FEATURE_DIMENSION # Import the final calculated dimension
    )
    # Import functions and classes from other project files
    from data_preprocessing import smiles_to_graph_data, initialize_feature_scaling
    from gnn_model import PairedLOHCGNN # Assumes GAT+MaxPool model is in gnn_model.py
except ImportError as e:
    print(f"--- CRITICAL ERROR ---")
    print(f"Failed to import required modules or config variables: {e}")
    print("Please ensure:")
    print("1. `config.py`, `data_preprocessing.py`, `gnn_model.py` are in the SAME directory as `train.py`.")
    print("2. These files do not contain syntax errors.")
    print("3. All names listed in the `from config import ...` line (like TOTAL_FEATURE_DIMENSION) are defined in `config.py`.")
    print("----------------------")
    exit(1) # Exit if imports fail
# ------------------------------------

# --- Setup Logging ---
logging.basicConfig(level=getattr(logging, LOGGING_LEVEL.upper(), logging.INFO), format=LOG_FORMAT)
# ---------------------

# Type alias for paired data
PairedDataTuple = Tuple[Data, Data]

# ================================================================
# Function Definitions (Moved BEFORE main())
# ================================================================

def load_and_preprocess_paired_data(data_path: str,
                                    smiles_de_col: str,
                                    smiles_hy_col: str,
                                    label_cols: List[str]
                                    ) -> Tuple[List[PairedDataTuple], int]:
    """
    Loads paired SMILES data, preprocesses into graph pairs using
    smiles_to_graph_data (with enhanced features),
    and initializes feature scaling.

    Args:
        data_path (str): Path to the Excel data file.
        smiles_de_col (str): Column name for dehydrogenated SMILES.
        smiles_hy_col (str): Column name for hydrogenated SMILES.
        label_cols (List[str]): List of target label column names.

    Returns:
        Tuple[List[PairedDataTuple], int]:
            - List of (graph_dehydro, graph_hydro) tuples.
            - The determined node feature dimension (should match TOTAL_FEATURE_DIMENSION).
        Returns ([], -1) on critical failure.
    """
    logging.info(f"Loading and preprocessing paired data from: {data_path} for labels: {label_cols}")
    if not os.path.exists(data_path):
        logging.error(f"Data file not found at {data_path}")
        return [], -1
    try:
        df = pd.read_excel(data_path)
    except Exception as e:
        logging.error(f"Error reading Excel file {data_path}: {e}")
        return [], -1

    required_cols = [smiles_de_col, smiles_hy_col] + label_cols
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logging.error(f"Required columns missing in {data_path}. Expected: {required_cols}, Missing: {missing}, Found: {list(df.columns)}")
        return [], -1

    try:
        feature_dimension = TOTAL_FEATURE_DIMENSION
        logging.info(f"Using total feature dimension from config: {feature_dimension}")
    except NameError:
         logging.error("FATAL: TOTAL_FEATURE_DIMENSION not found in config.py. Check config definition.")
         return [], -1

    try:
        initialize_feature_scaling(feature_dimension)
    except NameError:
         logging.error("Error: `initialize_feature_scaling` function not found. Check import from data_preprocessing.py.")
         return [], -1
    except ValueError as e:
        logging.error(f"Failed to initialize feature scaling: {e}")
        return [], -1

    paired_graph_data_list: List[PairedDataTuple] = []
    processed_count = 0
    skipped_count = 0
    original_indices_processed = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing SMILES pairs"):
        smiles_de = row[smiles_de_col]
        smiles_hy = row[smiles_hy_col]

        if not isinstance(smiles_de, str) or not smiles_de or \
           not isinstance(smiles_hy, str) or not smiles_hy:
            skipped_count += 1
            continue

        try:
            labels = [float(x) for x in row[label_cols].tolist()]
            if pd.isna(labels).any():
                skipped_count += 1
                continue
        except (ValueError, TypeError):
            skipped_count += 1
            continue

        try:
            graph_dehydro = smiles_to_graph_data(smiles_de, labels)
            graph_hydro = smiles_to_graph_data(smiles_hy, labels)
        except NameError:
             logging.error("Error: `smiles_to_graph_data` not found. Check import from data_preprocessing.py.")
             return [], -1
        except Exception as e:
             skipped_count += 1
             continue

        if graph_dehydro and graph_hydro:
            graph_dehydro.original_index = index
            graph_dehydro.smiles = smiles_de
            graph_hydro.original_index = index
            graph_hydro.smiles = smiles_hy

            paired_graph_data_list.append((graph_dehydro, graph_hydro))
            processed_count += 1
            original_indices_processed.append(index)
        else:
            skipped_count += 1

    logging.info(f"Successfully processed {processed_count} pairs.")
    if skipped_count > 0:
        logging.warning(f"Skipped {skipped_count} rows due to invalid SMILES, labels, or processing errors.")
    if not paired_graph_data_list:
        logging.error("No valid paired data could be loaded or processed.")
        return [], -1

    return paired_graph_data_list, feature_dimension


def calculate_label_scaling_params(paired_data_list: List[PairedDataTuple],
                                   train_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates min/max for label scaling based on the training set."""
    if not train_indices: raise ValueError("Training indices list is empty.")
    if not paired_data_list: raise ValueError("Paired data list is empty.")

    train_labels_list = [paired_data_list[i][0].y.numpy() for i in train_indices if paired_data_list[i][0].y is not None]

    if not train_labels_list: raise ValueError("Could not extract labels from the training set.")

    train_labels_np = np.concatenate(train_labels_list, axis=0)

    if np.isnan(train_labels_np).any():
        logging.warning("NaNs found in training labels during scaling calculation. Using nanmin/nanmax.")
        label_min_values = np.nanmin(train_labels_np, axis=0)
        label_max_values = np.nanmax(train_labels_np, axis=0)
    else:
        label_min_values = train_labels_np.min(axis=0)
        label_max_values = train_labels_np.max(axis=0)

    label_range = label_max_values - label_min_values
    zero_range_mask = np.abs(label_range) < 1e-9
    if np.any(zero_range_mask):
        logging.warning(f"Label columns {np.where(zero_range_mask)[0]} have zero range in training set. Setting range to 1.0 for scaling.")
        label_range[zero_range_mask] = 1.0

    logging.info(f"Calculated Label Scaling Min: {label_min_values}")
    logging.info(f"Calculated Label Scaling Max: {label_max_values}")
    return label_min_values, label_max_values


def apply_label_scaling(paired_data_list: List[PairedDataTuple],
                        label_min_values: np.ndarray,
                        label_max_values: np.ndarray):
    """Applies Min-Max scaling to labels in-place for all data points."""
    label_range = label_max_values - label_min_values
    zero_range_mask = np.abs(label_range) < 1e-9
    label_range[zero_range_mask] = 1.0

    scaled_count = 0
    for graph_de, graph_hy in paired_data_list:
        if graph_de.y is not None:
            y_np = graph_de.y.numpy()
            scaled_y_np = (y_np - label_min_values) / label_range
            scaled_y_tensor = torch.tensor(scaled_y_np, dtype=torch.float)
            graph_de.y = scaled_y_tensor
            graph_hy.y = scaled_y_tensor.clone()
            scaled_count += 1
    logging.info(f"Applied label scaling to {scaled_count} data pairs.")


def inverse_scale_labels(scaled_labels: np.ndarray,
                         min_vals: np.ndarray,
                         max_vals: np.ndarray) -> np.ndarray:
    """Converts scaled labels back to their original scale."""
    range_vals = max_vals - min_vals
    zero_range_mask = np.abs(range_vals) < 1e-9
    range_vals[zero_range_mask] = 1.0
    original_labels = scaled_labels * range_vals + min_vals
    return original_labels


def create_paired_dataloaders(all_scaled_data: List[PairedDataTuple],
                              train_indices: List[int],
                              test_indices: List[int],
                              batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Creates PyTorch Geometric DataLoaders for training and testing."""
    train_data = [all_scaled_data[i] for i in train_indices]
    test_data = [all_scaled_data[i] for i in test_indices]

    if not train_data: raise ValueError("Training data subset is empty after indexing.")
    if not test_data: logging.warning("Test data subset is empty after indexing.")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)

    logging.info(f"Created DataLoaders: Train batches={len(train_loader)}, Test batches={len(test_loader)}")
    return train_loader, test_loader


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device) -> float:
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0.0
    batch_count = 0
    for data_pair in tqdm(loader, desc="Training", leave=False):
        if isinstance(data_pair, (list, tuple)) and len(data_pair) == 2:
            batch_dehydro, batch_hydro = data_pair
        else:
            logging.warning(f"Unexpected data format from DataLoader: type={type(data_pair)}. Skipping batch.")
            continue

        batch_dehydro = batch_dehydro.to(device)
        batch_hydro = batch_hydro.to(device)

        optimizer.zero_grad()

        try:
            output = model(batch_dehydro, batch_hydro)
            target = batch_dehydro.y.view_as(output)

            if torch.isnan(output).any() or torch.isnan(target).any():
                 logging.warning("NaN detected in model output or target before loss calculation. Skipping batch.")
                 continue

            loss = criterion(output, target)

            if torch.isnan(loss):
                logging.warning("NaN loss detected. Skipping backward pass for this batch.")
                continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        except Exception as e:
            logging.error(f"Error during training batch: {e}", exc_info=True)
            continue

    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    return avg_loss


def evaluate_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> Tuple[float, np.ndarray, np.ndarray, List[Any]]:
    """
    Evaluates the model on the given data loader.
    Returns average loss, scaled predictions, scaled actuals, and identifiers.
    """
    model.eval()
    total_loss = 0.0
    predictions_scaled_list = []
    actual_values_scaled_list = []
    identifiers_list = []
    batch_count = 0

    with torch.no_grad():
        for data_pair in tqdm(loader, desc="Evaluating", leave=False):
            if isinstance(data_pair, (list, tuple)) and len(data_pair) == 2:
                batch_dehydro, batch_hydro = data_pair
            else:
                logging.warning(f"Unexpected data format from DataLoader: type={type(data_pair)}. Skipping batch.")
                continue

            batch_dehydro = batch_dehydro.to(device)
            batch_hydro = batch_hydro.to(device)

            try:
                num_in_batch = batch_dehydro.num_graphs
                if hasattr(batch_dehydro, 'smiles') and batch_dehydro.smiles is not None:
                    identifiers = batch_dehydro.smiles
                    if not isinstance(identifiers, list) or len(identifiers) != num_in_batch:
                        identifiers = [f"Index_{idx}" for idx in batch_dehydro.original_index.cpu().tolist()] if hasattr(batch_dehydro, 'original_index') else ["N/A"] * num_in_batch
                elif hasattr(batch_dehydro, 'original_index'):
                    idx_tensor = batch_dehydro.original_index
                    identifiers = idx_tensor.cpu().tolist() if isinstance(idx_tensor, torch.Tensor) else [idx_tensor] * num_in_batch
                    if not isinstance(identifiers, list) or len(identifiers) != num_in_batch:
                         identifiers = ["N/A"] * num_in_batch
                    else:
                         identifiers = [f"Index_{idx}" for idx in identifiers]
                else:
                    identifiers = ["N/A"] * num_in_batch

                if len(identifiers) != num_in_batch:
                     identifiers = (identifiers + ["Padding"] * num_in_batch)[:num_in_batch]

            except Exception as e:
                 num_in_batch = batch_dehydro.num_graphs if hasattr(batch_dehydro, 'num_graphs') else 0
                 logging.warning(f"Could not retrieve identifiers from batch: {e}")
                 identifiers = ["Error"] * num_in_batch

            try:
                output = model(batch_dehydro, batch_hydro)
                target = batch_dehydro.y.view_as(output)
                output_dim = output.shape[1]

                if torch.isnan(output).any() or torch.isnan(target).any():
                    logging.warning("NaN detected in output/target during evaluation.")
                    output_np = np.full((num_in_batch, output_dim), np.nan)
                    target_np = np.full((num_in_batch, output_dim), np.nan)
                else:
                    loss = criterion(output, target)
                    if not torch.isnan(loss):
                        total_loss += loss.item()
                    else:
                        logging.warning("NaN loss detected during evaluation.")
                    output_np = output.cpu().numpy()
                    target_np = target.cpu().numpy()

                predictions_scaled_list.append(output_np)
                actual_values_scaled_list.append(target_np)
                identifiers_list.extend(identifiers)
                batch_count += 1

            except Exception as e:
                logging.error(f"Error during evaluation batch: {e}", exc_info=True)
                num_in_batch = batch_dehydro.num_graphs if hasattr(batch_dehydro, 'num_graphs') else 0
                output_dim = model.mlp[-1].out_features
                predictions_scaled_list.append(np.full((num_in_batch, output_dim), np.nan))
                actual_values_scaled_list.append(np.full((num_in_batch, output_dim), np.nan))
                identifiers_list.extend(["EvalError"] * num_in_batch)
                continue

    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    predictions_scaled = np.concatenate(predictions_scaled_list, axis=0) if predictions_scaled_list else np.array([])
    actual_values_scaled = np.concatenate(actual_values_scaled_list, axis=0) if actual_values_scaled_list else np.array([])

    return avg_loss, predictions_scaled, actual_values_scaled, identifiers_list


def evaluate_metrics(predictions_original: np.ndarray, actual_values_original: np.ndarray,
                     property_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Calculates and logs evaluation metrics (MSE, R2) on original scale."""
    results = {}
    if predictions_original.size == 0 or actual_values_original.size == 0:
        logging.warning("Cannot evaluate metrics: No valid predictions/actuals provided.")
        return results
    if predictions_original.shape != actual_values_original.shape:
         logging.error(f"Shape mismatch for metric evaluation: Predictions {predictions_original.shape} vs Actuals {actual_values_original.shape}.")
         return results

    num_properties_to_eval = predictions_original.shape[1]
    if num_properties_to_eval != len(property_names):
         logging.warning(f"Number of predicted properties ({num_properties_to_eval}) does not match number of property names ({len(property_names)}). Using indices.")
         # Ensure property_names list matches the number of output columns
         property_names_used = [f"Property_{i}" for i in range(num_properties_to_eval)]
    else:
        property_names_used = property_names


    logging.info("\n--- Model Evaluation Metrics (Original Scale) ---")
    for i in range(num_properties_to_eval):
        prop_name = property_names_used[i]
        actual = actual_values_original[:, i]
        pred = predictions_original[:, i]

        valid_mask = ~np.isnan(actual) & ~np.isnan(pred)
        actual_valid = actual[valid_mask]
        pred_valid = pred[valid_mask]

        if actual_valid.size < 2:
            logging.warning(f"Skipping evaluation for '{prop_name}': Insufficient valid data points ({actual_valid.size}).")
            results[prop_name] = {'MSE': np.nan, 'R2': np.nan}
            continue

        try:
            mse = mean_squared_error(actual_valid, pred_valid)
            r2 = r2_score(actual_valid, pred_valid)
            results[prop_name] = {'MSE': mse, 'R2': r2}
            logging.info(f"  Property: {prop_name}")
            logging.info(f"    MSE : {mse:.4f}")
            logging.info(f"    R2  : {r2:.4f}")
        except Exception as e:
            logging.error(f"Error calculating metrics for {prop_name}: {e}")
            results[prop_name] = {'MSE': np.nan, 'R2': np.nan}

    logging.info("-------------------------------------------------")
    return results


def plot_results(train_losses: List[float], test_losses: List[float],
                 actual_original: np.ndarray, pred_original: np.ndarray,
                 property_names: List[str], output_dir: str = "."):
    """
    Generates loss curve and prediction vs actual scatter plots
    with specified formatting.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(MODEL_SAVE_PATH))[0]

    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.edgecolor': 'black', 'axes.labelcolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black',
        'axes.linewidth': 1.5, 'axes.titlecolor': 'black',
        'grid.color': 'lightgray', 'grid.linestyle': '--', 'grid.linewidth': 0.5,
        'legend.facecolor': 'white', 'legend.edgecolor': 'black', 'legend.labelcolor': 'black'
    })

    try:
        fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
        ax_loss.plot(train_losses, label='Train Loss', color='royalblue', linewidth=2)
        ax_loss.plot(test_losses, label='Test Loss', color='darkorange', linewidth=2)
        ax_loss.set_xlabel('Epoch', fontsize=12)
        ax_loss.set_ylabel('Loss (Scaled MSE)', fontsize=12)
        ax_loss.set_title(f'Training & Test Loss Curve ({base_filename})', fontsize=14)
        ax_loss.legend(fontsize=10)
        ax_loss.grid(True)
        ax_loss.tick_params(axis='both', which='major', labelsize=10, direction='in')
        ax_loss.set_ylim(bottom=0)
        loss_curve_path = os.path.join(output_dir, f'loss_curve_{base_filename}.png')
        fig_loss.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
        plt.close(fig_loss)
        logging.info(f"Loss curve saved to: {loss_curve_path}")
    except Exception as e:
        logging.error(f"Error plotting loss curve: {e}")
        plt.close()

    if pred_original.size == 0 or actual_original.size == 0 or pred_original.shape != actual_original.shape:
        logging.warning("Skipping scatter plots due to invalid or mismatched prediction/actual data.")
        return

    num_properties_to_plot = pred_original.shape[1]
    
    # Ensure property_names list matches the number of output columns for plotting
    if num_properties_to_plot != len(property_names):
        logging.warning(f"Mismatch between predicted properties ({num_properties_to_plot}) and names ({len(property_names)}) for plotting. Using generic names.")
        property_names_plot = [f"Target_{i+1}" for i in range(num_properties_to_plot)]
    else:
        property_names_plot = property_names

    for i in range(num_properties_to_plot):
        prop_name = property_names_plot[i]
        actual = actual_original[:, i]
        pred = pred_original[:, i]

        valid_mask = ~np.isnan(actual) & ~np.isnan(pred)
        actual_valid = actual[valid_mask]
        pred_valid = pred[valid_mask]

        if actual_valid.size == 0:
            logging.warning(f"Skipping scatter plot for '{prop_name}': No valid data points after NaN filtering.")
            continue

        try:
            fig_scatter, ax_scatter = plt.subplots(figsize=(6, 6))
            ax_scatter.scatter(actual_valid, pred_valid, alpha=0.7, s=30, color='deepskyblue', edgecolors='none')
            ax_scatter.set_xlabel(f'Actual {prop_name}', fontsize=14, fontweight='bold')
            ax_scatter.set_ylabel(f'Predicted {prop_name}', fontsize=14, fontweight='bold')
            ax_scatter.set_title(f'{prop_name}: Prediction vs Actual ({base_filename})', fontsize=16, fontweight='bold')
            ax_scatter.tick_params(axis='both', which='major', direction='out', labelsize=12)
            for ticklabel in ax_scatter.get_xticklabels(): ticklabel.set_fontweight('bold')
            for ticklabel in ax_scatter.get_yticklabels(): ticklabel.set_fontweight('bold')

            min_val = min(np.min(actual_valid), np.min(pred_valid))
            max_val = max(np.max(actual_valid), np.max(pred_valid))
            padding = (max_val - min_val) * 0.05
            plot_min = min_val - padding
            plot_max = max_val + padding
            ax_scatter.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=1.5, label='y=x')
            ax_scatter.set_xlim(plot_min, plot_max)
            ax_scatter.set_ylim(plot_min, plot_max)
            ax_scatter.legend(fontsize=10)
            ax_scatter.grid(False)
            fig_scatter.tight_layout()
            scatter_path = os.path.join(output_dir, f'{prop_name}_scatter_{base_filename}.png')
            fig_scatter.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close(fig_scatter)
            logging.info(f"Scatter plot for {prop_name} saved to: {scatter_path}")
        except Exception as e:
            logging.error(f"Error plotting scatter for {prop_name}: {e}")
            plt.close()


def main():
    """Main function to run the training and evaluation pipeline."""
    try:
        logging.info(f"--- Starting LOHC GNN Training Pipeline ({os.path.basename(MODEL_SAVE_PATH)}) ---")
        logging.info(f"Predicting target(s): {PROPERTY_NAMES}") # Will now list multiple targets

        paired_data_list, num_node_features = load_and_preprocess_paired_data(
            DATA_FILE_PATH, DEHYDRO_SMILES_COL, HYDRO_SMILES_COL, LABEL_COLS
        )
        if not paired_data_list or num_node_features <= 0:
            logging.error("Exiting: Failed to load or preprocess data.")
            return
        logging.info(f"Loaded {len(paired_data_list)} data pairs. Node feature dimension: {num_node_features}")
        if num_node_features != TOTAL_FEATURE_DIMENSION:
             logging.warning(f"Returned feature dimension ({num_node_features}) doesn't match config ({TOTAL_FEATURE_DIMENSION}). Check data processing logic.")

        num_samples = len(paired_data_list)
        indices = list(range(num_samples))
        train_indices, test_indices = train_test_split(
            indices,
            test_size=HYPERPARAMS['test_split_ratio'],
            random_state=HYPERPARAMS['random_state']
        )
        logging.info(f"Data split: {len(train_indices)} train, {len(test_indices)} test samples.")

        label_min_values, label_max_values = calculate_label_scaling_params(paired_data_list, train_indices)
        apply_label_scaling(paired_data_list, label_min_values, label_max_values)

        train_loader, test_loader = create_paired_dataloaders(
            paired_data_list, train_indices, test_indices, HYPERPARAMS['batch_size']
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        model = PairedLOHCGNN(
            num_node_features=num_node_features,
            hidden_channels=HYPERPARAMS['hidden_channels'],
            num_output_features=HYPERPARAMS['num_output_features'], # Now 3
            dropout_rate=HYPERPARAMS['dropout_rate'],
            gnn_layers=HYPERPARAMS.get('gnn_layers', 3),
            gat_heads=HYPERPARAMS.get('gat_heads', 4),
            gat_output_heads=HYPERPARAMS.get('gat_output_heads', 1)
        ).to(device)

        logging.info(f"Model initialized (Output Features: {HYPERPARAMS['num_output_features']}):")
        # logging.info(model)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS['learning_rate'])

        logging.info(f"Starting training for {HYPERPARAMS['epochs']} epochs...")
        train_losses, test_losses = [], []
        best_test_loss = float('inf')

        for epoch in range(HYPERPARAMS['epochs']):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, _, _, _ = evaluate_epoch(model, test_loader, criterion, device)
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if (epoch + 1) % 50 == 0 or epoch == 0:
                logging.info(f"Epoch {epoch+1}/{HYPERPARAMS['epochs']} | Train Loss (scaled): {train_loss:.4f} | Test Loss (scaled): {test_loss:.4f}")

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                try:
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    if (epoch + 1) % 10 == 0 or epoch == HYPERPARAMS['epochs'] - 1:
                         logging.debug(f"Epoch {epoch+1}: New best model saved (Test Loss: {best_test_loss:.4f})")
                except Exception as e:
                    logging.error(f"Error saving model checkpoint at epoch {epoch+1}: {e}")

        logging.info("Training finished.")
        logging.info(f"Best test loss achieved (scaled): {best_test_loss:.4f}")
        logging.info(f"Model weights saved to: {MODEL_SAVE_PATH}")

        logging.info("Performing final evaluation on test set with the best saved model...")
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            logging.info(f"Loaded best model weights from {MODEL_SAVE_PATH}")
        except FileNotFoundError:
             logging.error(f"Could not load best model weights. File not found: {MODEL_SAVE_PATH}")
             logging.warning("Evaluating with the weights from the end of training instead.")
        except Exception as e:
            logging.error(f"Could not load best model weights from {MODEL_SAVE_PATH}. Error: {e}")
            logging.warning("Evaluating with the weights from the end of training instead.")

        final_test_loss, predictions_scaled, actual_values_scaled, test_identifiers = evaluate_epoch(
            model, test_loader, criterion, device
        )
        logging.info(f"Final Test Loss (Scaled MSE) with loaded model: {final_test_loss:.4f}")

        predictions_original = inverse_scale_labels(predictions_scaled, label_min_values, label_max_values)
        actual_values_original = inverse_scale_labels(actual_values_scaled, label_min_values, label_max_values)

        evaluate_metrics(predictions_original, actual_values_original, PROPERTY_NAMES)

        logging.info("\n--- Analyzing Outliers (Large Prediction Errors on Test Set for the first property) ---")
        try:
            # Outlier analysis for the first property in PROPERTY_NAMES
            # This can be extended to loop through all properties if needed
            if predictions_original.ndim >= 2 and actual_values_original.ndim >=2 and \
               predictions_original.shape[1] > 0 and actual_values_original.shape[1] > 0:

                pred_col_idx = 0 # Analyze the first predicted property
                actual_col_idx = 0 # Corresponding actual property

                pred_flat = predictions_original[:, pred_col_idx]
                actual_flat = actual_values_original[:, actual_col_idx]
                current_property_name = PROPERTY_NAMES[pred_col_idx] if pred_col_idx < len(PROPERTY_NAMES) else f"Property_{pred_col_idx}"


                if len(pred_flat) == len(actual_flat) == len(test_identifiers):
                    valid_mask_outlier = ~np.isnan(actual_flat) & ~np.isnan(pred_flat)
                    actual_flat_valid = actual_flat[valid_mask_outlier]
                    pred_flat_valid = pred_flat[valid_mask_outlier]
                    identifiers_valid = [test_identifiers[i] for i, valid in enumerate(valid_mask_outlier) if valid]

                    if len(actual_flat_valid) > 1:
                        errors = np.abs(pred_flat_valid - actual_flat_valid)
                        error_threshold = np.std(actual_flat_valid) * 1.5 # Example: 1.5 std dev

                        outliers_info = []
                        for i in range(len(identifiers_valid)):
                            if errors[i] > error_threshold:
                                outliers_info.append({
                                    'identifier': identifiers_valid[i],
                                    'actual': actual_flat_valid[i],
                                    'predicted': pred_flat_valid[i],
                                    'error': errors[i]
                                })
                        outliers_info.sort(key=lambda x: x['error'], reverse=True)

                        if outliers_info:
                            logging.info(f"Found {len(outliers_info)} outliers for '{current_property_name}' with absolute error > {error_threshold:.4f}:")
                            for i, outlier in enumerate(outliers_info[:10]):
                                logging.info(f"  #{i+1}: ID={outlier['identifier']}, Actual={outlier['actual']:.4f}, Pred={outlier['predicted']:.4f}, Err={outlier['error']:.4f}") # Adjusted precision for predicted
                        else:
                            logging.info(f"No outliers found for '{current_property_name}' with absolute error > {error_threshold:.4f}.")
                    else:
                         logging.warning(f"Not enough valid data points for '{current_property_name}' to perform outlier analysis.")
                else:
                     logging.warning(f"Length mismatch during outlier analysis for '{current_property_name}'. Skipping.")
            else:
                logging.warning("Prediction or actual data is not 2D or has no columns for outlier analysis. Skipping.")


        except Exception as e:
            logging.error(f"Error during outlier analysis: {e}", exc_info=True)
        logging.info("-------------------------------------------------------------")

        logging.info("Generating final plots...")
        plot_results(train_losses, test_losses, actual_values_original, predictions_original, PROPERTY_NAMES)

        logging.info("Saving label scaling parameters...")
        try:
            np.savez(LABEL_SCALING_PARAMS_PATH, min_vals=label_min_values, max_vals=label_max_values)
            logging.info(f"Label scaling parameters saved to {LABEL_SCALING_PARAMS_PATH}")
        except Exception as e:
            logging.error(f"Error saving label scaling parameters: {e}")

        logging.info("--- LOHC GNN Training Pipeline Complete ---")

    except NameError as ne:
        logging.error(f"--- NameError Occurred ---")
        logging.error(f"A variable or function name is likely not defined or imported correctly: {ne}")
        logging.error("Check imports (especially from config.py), variable definitions, and function calls.")
        logging.error("--------------------------")
    except Exception as ex:
        logging.error(f"--- An Unexpected Error Occurred During Pipeline Execution ---", exc_info=True)


if __name__ == "__main__":
    main()
# --- END OF FILE train.py ---