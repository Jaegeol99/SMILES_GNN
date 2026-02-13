import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import os
import logging

from config import (
    HYPERPARAMS, PROPERTY_NAMES, MODEL_SAVE_PATH, LABEL_SCALING_PARAMS_PATH, GLOBAL_SCALING_PARAMS_PATH, LOGGING_LEVEL, LOG_FORMAT
)
from data_processing import (
    load_and_preprocess_qm9_data,
    calculate_label_scaling_params,
    apply_label_scaling,
    inverse_scale_labels,
    calculate_global_scaling_params,
    apply_global_scaling,
    create_dataloaders,
)
from gnn_model import LOHCGNN
from training_utils import (
    train_epoch, evaluate_epoch, evaluate_metrics, plot_results
)
from feature_configs import GLOBAL_FEATURE_DIM, NUM_BOND_FEATURES, NUM_LINE_EDGE_FEATURES, LINE_NODE_FEATURE_DIM

def main():
    logging.basicConfig(level=LOGGING_LEVEL, format=LOG_FORMAT)
    
    data_list, num_node_features = load_and_preprocess_qm9_data()
    if not data_list:
        logging.error("No valid data loaded. Exiting.")
        return

    indices = list(range(len(data_list)))
    
    # 데이터를 80% 훈련, 20% 임시 세트로 분할
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=HYPERPARAMS['test_split_ratio'],
        random_state=HYPERPARAMS['random_state']
    )
    
    # 임시 세트를 50% 검증, 50% 테스트 세트로 분할 (전체의 10%씩)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        random_state=HYPERPARAMS['random_state']
    )

    logging.info(f"Data split: {len(train_indices)} train, {len(val_indices)} validation, {len(test_indices)} test samples.")

    label_median, label_iqr = calculate_label_scaling_params(
        data_list, train_indices
    )

    if label_median is not None and label_iqr is not None:
        np.savez(LABEL_SCALING_PARAMS_PATH, median=label_median, iqr=label_iqr)
        apply_label_scaling(data_list, label_median, label_iqr)

    # Fit global descriptor scaler on TRAIN split only, then apply to all splits.
    g_mean, g_std = calculate_global_scaling_params(data_list, train_indices)
    np.savez(GLOBAL_SCALING_PARAMS_PATH, mean=g_mean, std=g_std)
    apply_global_scaling(
        data_list,
        g_mean,
        g_std,
        clip_value=HYPERPARAMS.get('global_clip_value', None),
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        data_list, train_indices, val_indices, test_indices, HYPERPARAMS["batch_size"]
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    model = LOHCGNN(
        node_in_dim=num_node_features,
        edge_in_dim=NUM_BOND_FEATURES,
        line_node_in_dim=LINE_NODE_FEATURE_DIM,
        line_edge_in_dim=NUM_LINE_EDGE_FEATURES,
        hidden_dim=HYPERPARAMS['hidden_dim'],
        num_layers=HYPERPARAMS['num_layers'],
        num_output_features=HYPERPARAMS['num_output_features'],
        dropout_rate=HYPERPARAMS['dropout_rate'],
        global_in_dim=GLOBAL_FEATURE_DIM,
    ).to(device)

    # -------------------------
    # Loss
    # -------------------------
    loss_type = str(HYPERPARAMS.get('loss_type', 'mse')).lower()
    if loss_type == 'huber':
        delta = float(HYPERPARAMS.get('huber_delta', 1.0))
        # Prefer nn.HuberLoss when available; fallback to SmoothL1Loss(beta=delta)
        if hasattr(nn, 'HuberLoss'):
            criterion = nn.HuberLoss(delta=delta)
        else:
            try:
                criterion = nn.SmoothL1Loss(beta=delta)
            except TypeError:
                # Older PyTorch
                criterion = nn.SmoothL1Loss()
        logging.info(f"Using Huber-style loss with delta={delta}")
    else:
        criterion = nn.MSELoss()
        logging.info("Using MSE loss")
    optimizer = optim.AdamW(model.parameters(), lr=HYPERPARAMS['learning_rate'], weight_decay=HYPERPARAMS.get('weight_decay', 0.0))

    scheduler = None
    if HYPERPARAMS.get('use_scheduler', False):
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=HYPERPARAMS.get('lr_factor', 0.5),
            patience=HYPERPARAMS.get('lr_patience', 5),
            min_lr=HYPERPARAMS.get('min_lr', 1e-6),
            verbose=True,
        )


    best_val_mae = float('inf')
    train_losses_history = []
    val_losses_history = []

    for epoch in range(HYPERPARAMS['epochs']):
        train_out = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip_norm=HYPERPARAMS.get('grad_clip_norm', None),
            log_grad_norm=bool(HYPERPARAMS.get('log_grad_norm', False)),
        )
        train_loss = float(train_out.get('loss', 0.0))

        val_loss, val_pred_scaled, val_true_scaled, _ = evaluate_epoch(model, val_loader, criterion, device)

        val_pred_scaled = np.asarray(val_pred_scaled)
        val_true_scaled = np.asarray(val_true_scaled)

        if val_pred_scaled.size == 0 or val_true_scaled.size == 0:
            logging.info(
                f"Epoch {epoch+1}/{HYPERPARAMS['epochs']}: "
                f"Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f} (No valid predictions to evaluate)"
            )
            continue

        val_pred = val_pred_scaled
        val_true = val_true_scaled
        if label_median is not None and label_iqr is not None:
            val_pred = inverse_scale_labels(val_pred_scaled, label_median, label_iqr)
            val_true = inverse_scale_labels(val_true_scaled, label_median, label_iqr)

        r2_per = r2_score(val_true, val_pred, multioutput='raw_values')
        mae_per = mean_absolute_error(val_true, val_pred, multioutput='raw_values')
        r2_mean = float(np.mean(r2_per))
        mae_mean = float(np.mean(mae_per))

        msg = (
            f"Epoch {epoch+1}/{HYPERPARAMS['epochs']}: "
            f"Train Loss = {train_loss:.6f}, Validation Loss = {val_loss:.6f}, "
            f"Validation R2 = {r2_mean:.6f}, Validation MAE = {mae_mean:.6f}"
        )
        if bool(HYPERPARAMS.get('log_grad_norm', False)) or (HYPERPARAMS.get('grad_clip_norm', None) not in (None, 0)):
            gmean = train_out.get('grad_norm_mean', None)
            gmax = train_out.get('grad_norm_max', None)
            if gmean is not None and gmax is not None:
                msg += f", GradNorm(mean/max) = {float(gmean):.4g}/{float(gmax):.4g}"
        logging.info(msg)

        # LR scheduling (optional)
        if scheduler is not None:
            metric = mae_mean if HYPERPARAMS.get('scheduler_metric','mae') == 'mae' else val_loss
            scheduler.step(metric)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Current LR: {current_lr:.6g}")

        train_losses_history.append(train_loss)
        val_losses_history.append(val_loss)

        # NOTE: Optimizing MSE (val_loss) does not necessarily optimize MAE.
        # If your goal metric is MAE, checkpoint on MAE instead.
        if mae_mean < best_val_mae:
            best_val_mae = mae_mean
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logging.info(f"Saved best model with validation MAE: {best_val_mae:.6f}")

    if len(test_loader.dataset) > 0:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        logging.info("Loaded best model for final evaluation on the test set")
        
        _, predictions_scaled, actual_values_scaled, _ = evaluate_epoch(
            model, test_loader, criterion, device
        )

        if predictions_scaled.size > 0 and actual_values_scaled.size > 0:
            predictions_original = predictions_scaled
            actual_values_original = actual_values_scaled
            if label_median is not None and label_iqr is not None:
                predictions_original = inverse_scale_labels(
                    predictions_scaled, label_median, label_iqr
                    )
                actual_values_original = inverse_scale_labels(
                    actual_values_scaled, label_median, label_iqr
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