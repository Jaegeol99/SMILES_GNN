# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import logging

from config import *
from data_processing import *
from gnn_model import LOHCGNN
from training_utils import *
from feature_configs import NUM_BOND_FEATURES, NUM_LINE_EDGE_FEATURES

def main():
    logging.basicConfig(level=LOGGING_LEVEL, format=LOG_FORMAT)
    paired_data, node_dim = load_and_preprocess_paired_data()
    
    indices = list(range(len(paired_data)))
    tr_idx, temp_idx = train_test_split(indices, test_size=HYPERPARAMS['test_split_ratio'], random_state=HYPERPARAMS['random_state'])
    val_idx, te_idx = train_test_split(temp_idx, test_size=0.5, random_state=HYPERPARAMS['random_state'])

    median, iqr = calculate_label_scaling_params(paired_data, tr_idx)
    np.savez(LABEL_SCALING_PARAMS_PATH, median=median, iqr=iqr)
    apply_label_scaling(paired_data, median, iqr)

    tr_loader, val_loader, te_loader = create_paired_dataloaders(paired_data, tr_idx, val_idx, te_idx, HYPERPARAMS['batch_size'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LOHCGNN(node_dim, NUM_BOND_FEATURES, NUM_LINE_EDGE_FEATURES, 
                    HYPERPARAMS['hidden_dim'], HYPERPARAMS['num_layers'], 
                    HYPERPARAMS['num_output_features'], HYPERPARAMS['dropout_rate']).to(device)

    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS['learning_rate'])
    criterion = nn.MSELoss()

    best_val = float('inf')
    tr_hist, val_hist = [], []

    for epoch in range(HYPERPARAMS['epochs']):
        tr_l = train_epoch(model, tr_loader, criterion, optimizer, device)
        val_l, _, _, _ = evaluate_epoch(model, val_loader, criterion, device)
        tr_hist.append(tr_l); val_hist.append(val_l)
        if val_l < best_val:
            best_val = val_l
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        logging.info(f"Epoch {epoch+1}: Tr {tr_l:.4f}, Val {val_l:.4f}")

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    _, p_sc, a_sc, _ = evaluate_epoch(model, te_loader, criterion, device)
    p_orig, a_orig = inverse_scale_labels(p_sc, median, iqr), inverse_scale_labels(a_sc, median, iqr)
    metrics = evaluate_metrics(p_orig, a_orig, PROPERTY_NAMES)
    plot_results(tr_hist, val_hist, a_orig, p_orig, PROPERTY_NAMES, metrics)

if __name__ == "__main__":
    main()