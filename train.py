import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import os

from config import (
    HYPERPARAMS, PROPERTY_NAMES, MODEL_SAVE_PATH, LABEL_SCALING_PARAMS_PATH
)
from data_processing import (
    load_and_preprocess_paired_data,
    calculate_label_scaling_params,
    apply_label_scaling,
    inverse_scale_labels,
    create_paired_dataloaders
)
from gnn_model import PairedLOHCGNN
from training_utils import (
    train_epoch, evaluate_epoch, evaluate_metrics, plot_results
)

def main():
    paired_data_list, num_node_features = load_and_preprocess_paired_data()

    if not paired_data_list or num_node_features <= 0:
        return

    indices = list(range(len(paired_data_list)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=HYPERPARAMS['test_split_ratio'],
        random_state=HYPERPARAMS['random_state']
    )

    label_min_values, label_max_values = calculate_label_scaling_params(
        paired_data_list, train_indices
    )

    if label_min_values is not None and label_max_values is not None:
        np.savez(LABEL_SCALING_PARAMS_PATH, min_vals=label_min_values, max_vals=label_max_values)
        apply_label_scaling(paired_data_list, label_min_values, label_max_values)

    train_loader, test_loader = create_paired_dataloaders(
        paired_data_list, train_indices, test_indices, HYPERPARAMS['batch_size']
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PairedLOHCGNN(
        num_node_features=num_node_features,
        hidden_channels=HYPERPARAMS['hidden_channels'],
        num_output_features=HYPERPARAMS['num_output_features'],
        dropout_rate=HYPERPARAMS['dropout_rate'],
        gnn_layers=HYPERPARAMS.get('gnn_layers', 3),
        gat_heads=HYPERPARAMS.get('gat_heads', 4),
        gat_output_heads=HYPERPARAMS.get('gat_output_heads', 1)
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS['learning_rate'])

    best_test_loss = float('inf')
    train_losses_history = []
    test_losses_history = []

    for epoch in range(HYPERPARAMS['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, _, _, _ = evaluate_epoch(model, test_loader, criterion, device)

        train_losses_history.append(train_loss)
        test_losses_history.append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    if len(test_loader.dataset) > 0:
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

            evaluate_metrics(predictions_original, actual_values_original, PROPERTY_NAMES)

            plot_results(
                train_losses_history,
                test_losses_history,
                actual_values_original,
                predictions_original,
                PROPERTY_NAMES,
                output_dir="."
            )

if __name__ == "__main__":
    main()