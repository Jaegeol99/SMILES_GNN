# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import logging

from config import *
from data_processing import *
from gnn_model import AtomicChargeGNN
from training_utils import *
from feature_configs import NUM_BOND_FEATURES

def main():
    logging.basicConfig(level=LOGGING_LEVEL, format=LOG_FORMAT)
    
    data_list, num_node_features = load_and_preprocess_data()
    if not data_list:
        logging.error("No data loaded. Exiting.")
        return

    indices = list(range(len(data_list)))
    train_indices, test_indices = train_test_split(indices, test_size=HYPERPARAMS['test_split_ratio'], random_state=HYPERPARAMS['random_state'])

    scaler = calculate_label_scaling_params(data_list, train_indices)
    if scaler:
        label_min, label_max = scaler
        np.savez(LABEL_SCALING_PARAMS_PATH, min_val=label_min, max_val=label_max)
        apply_label_scaling(data_list, label_min, label_max)

    train_loader, test_loader = create_dataloaders(data_list, train_indices, test_indices, HYPERPARAMS['batch_size'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    model = AtomicChargeGNN(
        node_in_dim=num_node_features,
        edge_in_dim=NUM_BOND_FEATURES,
        hidden_dim=HYPERPARAMS['hidden_dim'],
        num_layers=HYPERPARAMS['num_layers'],
        num_output_features=HYPERPARAMS['num_output_features'],
        dropout_rate=HYPERPARAMS['dropout_rate']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS['learning_rate'])

    best_test_loss = float('inf')
    train_losses, test_losses = [], []

    for epoch in range(HYPERPARAMS['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, _, _ = evaluate_epoch(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        logging.info(f"Epoch {epoch+1}/{HYPERPARAMS['epochs']}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logging.info(f"Saved best model with test loss: {best_test_loss:.6f}")

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    _, preds_scaled, actuals_scaled = evaluate_epoch(model, test_loader, criterion, device)

    if scaler:
        preds_orig = inverse_scale_labels(preds_scaled, scaler[0], scaler[1])
        actuals_orig = inverse_scale_labels(actuals_scaled, scaler[0], scaler[1])
        
        final_metrics = evaluate_metrics(preds_orig, actuals_orig, PROPERTY_NAMES[0])
        plot_results(train_losses, test_losses, actuals_orig, preds_orig, PROPERTY_NAMES[0], final_metrics)

    logging.info("Training and evaluation finished.")

if __name__ == "__main__":
    main()