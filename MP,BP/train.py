import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import os
import logging

from config import (
    HYPERPARAMS, PROPERTY_NAMES, MODEL_SAVE_PATH, LABEL_SCALING_PARAMS_PATH, LOGGING_LEVEL, LOG_FORMAT
)
from data_processing import (
    load_and_preprocess_data,
    calculate_label_scaling_params,
    apply_label_scaling,
    inverse_scale_labels,
    create_dataloaders
)
from gnn_model import LOHCGNN
from training_utils import (
    train_epoch, evaluate_epoch, evaluate_metrics, plot_results
)
from feature_configs import NUM_BOND_FEATURES, NUM_LINE_EDGE_FEATURES

def main():
    logging.basicConfig(level=LOGGING_LEVEL, format=LOG_FORMAT)
    
    data_list, num_node_features = load_and_preprocess_data()
    if not data_list:
        logging.error("No valid data loaded. Exiting.")
        return

    indices = list(range(len(data_list)))
    
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=HYPERPARAMS['test_split_ratio'],
        random_state=HYPERPARAMS['random_state']
    )
    
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        random_state=HYPERPARAMS['random_state']
    )
    logging.info(f"Data split: {len(train_indices)} train, {len(val_indices)} validation, {len(test_indices)} test samples.")

    label_min_values, label_max_values = calculate_label_scaling_params(
        data_list, train_indices
    )

    if label_min_values is not None and label_max_values is not None:
        np.savez(LABEL_SCALING_PARAMS_PATH, min_vals=label_min_values, max_vals=label_max_values)
        apply_label_scaling(data_list, label_min_values, label_max_values)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_list, train_indices, val_indices, test_indices, HYPERPARAMS['batch_size']
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    model = LOHCGNN(
        node_in_dim=num_node_features,
        edge_in_dim=NUM_BOND_FEATURES,
        line_edge_in_dim=NUM_LINE_EDGE_FEATURES,
        hidden_dim=HYPERPARAMS['hidden_dim'],
        num_layers=HYPERPARAMS['num_layers'],
        num_output_features=HYPERPARAMS['num_output_features'],
        dropout_rate=HYPERPARAMS['dropout_rate']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS['learning_rate'])

    best_val_loss = float('inf')
    train_losses_history = []
    val_losses_history = []

    for epoch in range(HYPERPARAMS['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _, _ = evaluate_epoch(model, val_loader, criterion, device)

        logging.info(f"Epoch {epoch+1}/{HYPERPARAMS['epochs']}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")
        train_losses_history.append(train_loss)
        val_losses_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logging.info(f"Saved best model with validation loss: {best_val_loss:.4f}")

    if len(test_loader.dataset) > 0:
        logging.info("Loading best model for final evaluation on the test set")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        
        _, predictions_scaled, actual_values_scaled, _ = evaluate_epoch(
            model, test_loader, criterion, device
        )

        if predictions_scaled.size > 0 and actual_values_scaled.size > 0:
            predictions_original = predictions_scaled
            actual_values_original = actual_values_scaled
            if label_min_values is not None and label_max_values is not None:
                predictions_original = inverse_scale_labels(
                    predictions_scaled, label_min_values, label_max_values
                )
                actual_values_original = inverse_scale_labels(
                    actual_values_scaled, label_min_values, label_max_values
                )

            final_metrics = evaluate_metrics(predictions_original, actual_values_original, PROPERTY_NAMES)

            plot_results(
                train_losses_history,
                val_losses_history,
                actual_values_original,
                predictions_original,
                PROPERTY_NAMES,
                metrics=final_metrics,
                output_dir="."
            )
            logging.info("Evaluation and plotting completed.")
        else:
            logging.warning("No predictions or actual values to evaluate on the test set.")
    else:
        logging.warning("Test dataset is empty.")

if __name__ == "__main__":
    main()