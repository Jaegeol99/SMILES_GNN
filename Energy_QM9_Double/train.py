import argparse
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from config import (
    ABLATION_RESULTS_PATH,
    CSV_INDEX_COL,
    CSV_PATH,
    CSV_SMILES_COL,
    GLOBAL_SCALING_PARAMS_PATH,
    HYPERPARAMS,
    LABEL_SCALING_PARAMS_PATH,
    LOG_FORMAT,
    LOGGING_LEVEL,
    MAX_SAMPLES,
    MODEL_SAVE_PATH,
    STAGE2_MODEL_SAVE_PATH,
    OUTPUT_DIR,
    PROPERTY_NAMES,
    REPORT_DIR,
    SEED,
)
from data_processing import (
    apply_global_scaling,
    apply_label_scaling,
    calculate_label_scaling_params,
    create_dataloaders,
    inverse_scale_labels,
    load_and_preprocess_qm9_data,
)
from feature_configs import (
    GLOBAL_FEATURE_DIM,
    LINE_NODE_FEATURE_DIM,
    NUM_BOND_FEATURES,
    NUM_LINE_EDGE_FEATURES,
)
from gnn_model import LOHCGNN
from training_utils import (
    compute_error_percentiles,
    evaluate_epoch,
    evaluate_metrics,
    get_worst_k_samples,
    plot_results,
    save_worst_k_csv,
    train_epoch,
)


def _load_id_to_smiles_map() -> Dict[str, str]:
    """Load a mapping {mol_id(str) -> SMILES(str)} from the input CSV.

    This is used only for reporting (worst-k CSV). We avoid storing SMILES
    inside PyG Data objects because string batching/collation is brittle.
    """
    try:
        import pandas as pd

        df = pd.read_csv(CSV_PATH)
        if MAX_SAMPLES is not None:
            df = df.iloc[:MAX_SAMPLES].copy()

        if CSV_INDEX_COL in df.columns:
            ids = df[CSV_INDEX_COL].astype(int).astype(str).tolist()
        else:
            ids = [str(int(i)) for i in range(len(df))]

        smiles = df[CSV_SMILES_COL].astype(str).tolist() if CSV_SMILES_COL in df.columns else ["" for _ in ids]
        return dict(zip(ids, smiles))
    except Exception:
        return {}


def _enrich_worst_rows_with_smiles(rows: List[Dict[str, Any]], id2smiles: Dict[str, str]) -> None:
    for rank, r in enumerate(rows, start=1):
        rid = str(r.get("id", ""))
        r["rank"] = rank
        r["smiles"] = id2smiles.get(rid, "")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep deterministic behavior where possible (can reduce throughput).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_global_standard_scaler(
    data_list: List[Tuple[Any, Any]],
    train_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit StandardScaler on TRAIN split global descriptors only.

    This prevents train/val/test leakage from global feature normalization.
    """
    xs: List[np.ndarray] = []
    for idx in train_indices:
        atom_g, _ = data_list[idx]
        if not hasattr(atom_g, "g") or atom_g.g is None:
            continue
        g = atom_g.g.detach().cpu().numpy().reshape(1, -1)
        xs.append(g)

    if not xs:
        mean = np.zeros((GLOBAL_FEATURE_DIM,), dtype=np.float32)
        std = np.ones((GLOBAL_FEATURE_DIM,), dtype=np.float32)
        return mean, std

    X = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X)
    mean = scaler.mean_.astype(np.float32, copy=False)
    std = scaler.scale_.astype(np.float32, copy=False)
    std = np.where(std < 1e-12, 1.0, std).astype(np.float32, copy=False)
    return mean, std


def build_model(num_node_features: int, hp: Dict[str, Any], device: torch.device) -> LOHCGNN:
    model = LOHCGNN(
        node_in_dim=num_node_features,
        edge_in_dim=NUM_BOND_FEATURES,  # kept for backward-compat; not used in A2/A3
        line_node_in_dim=LINE_NODE_FEATURE_DIM,
        line_edge_in_dim=NUM_LINE_EDGE_FEATURES,  # kept for backward-compat; not used in A2/A3
        hidden_dim=hp["hidden_dim"],
        num_layers=hp["num_layers"],
        num_output_features=hp["num_output_features"],
        dropout_rate=hp["dropout_rate"],
        global_in_dim=GLOBAL_FEATURE_DIM,
    ).to(device)
    return model


def build_loss(hp: Dict[str, Any]) -> nn.Module:
    loss_type = str(hp.get("loss_type", "mse")).lower()
    if loss_type == "huber":
        return nn.HuberLoss(delta=float(hp.get("huber_delta", 1.0)))
    return nn.MSELoss()


def build_optimizer(model: nn.Module, hp: Dict[str, Any]) -> optim.Optimizer:
    opt = str(hp.get("optimizer", "adamw")).lower()
    lr = float(hp["learning_rate"])
    wd = float(hp.get("weight_decay", 0.0))
    if opt == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def build_scheduler(optimizer: optim.Optimizer, hp: Dict[str, Any]) -> Optional[ReduceLROnPlateau]:
    sched = str(hp.get("scheduler", "none")).lower()
    if sched != "plateau":
        return None
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(hp.get("scheduler_factor", 0.5)),
        patience=int(hp.get("scheduler_patience", 5)),
        min_lr=float(hp.get("scheduler_min_lr", 1e-6)),
        verbose=False,
    )


def log_val_diagnostics(
    epoch: int,
    hp: Dict[str, Any],
    val_true: np.ndarray,
    val_pred: np.ndarray,
    identifiers: List[str],
    tracker: Optional[Dict[str, Any]] = None,
) -> float:
    """Log target-wise MAE + worst-k errors + distribution stats on validation set.

    Returns:
        mae_mean
    """
    r2_per = r2_score(val_true, val_pred, multioutput="raw_values")
    mae_per = mean_absolute_error(val_true, val_pred, multioutput="raw_values")
    r2_mean = float(np.mean(r2_per))
    mae_mean = float(np.mean(mae_per))

    if bool(hp.get("log_targetwise_mae", True)):
        per_str = ", ".join([f"{PROPERTY_NAMES[i]}: {float(mae_per[i]):.6f}" for i in range(len(mae_per))])
        logging.info(f"  Val MAE (per target): {per_str}")
        per_str_r2 = ", ".join([f"{PROPERTY_NAMES[i]}: {float(r2_per[i]):.4f}" for i in range(len(r2_per))])
        logging.info(f"  Val R2  (per target): {per_str_r2}")

    # Distribution stats
    abs_err = np.abs(val_pred - val_true)
    p = [int(x) for x in hp.get("error_percentiles", [50, 90, 95, 99])]
    stats_all = compute_error_percentiles(abs_err, p)
    logging.info("  Val |error| stats (all targets pooled): " + ", ".join([f"{k}={v:.6g}" for k, v in stats_all.items()]))

    # Sample-wise max(|err|) (helps detect outlier-dominated tails driving MAE)
    if abs_err.ndim == 2:
        max_per_sample = abs_err.max(axis=1)
    else:
        max_per_sample = abs_err
    stats_smax = compute_error_percentiles(max_per_sample, p)
    logging.info("  Val max(|error|) per-sample stats: " + ", ".join([f"{k}={v:.6g}" for k, v in stats_smax.items()]))

    # Persist percentile diagnostics for easy epoch-to-epoch inspection.
    if bool(hp.get("save_epoch_reports", True)):
        os.makedirs(REPORT_DIR, exist_ok=True)
        pct_csv = os.path.join(REPORT_DIR, "val_error_percentiles.csv")
        import csv

        # Build header dynamically (handles multi-target).
        base_cols = [
            "epoch",
            "pooled_mean", "pooled_max",
        ] + [f"pooled_{k}" for k in stats_all.keys() if k.startswith("p")] + [
            "smax_mean", "smax_max",
        ] + [f"smax_{k}" for k in stats_smax.keys() if k.startswith("p")]
        tgt_cols: List[str] = []
        if abs_err.ndim == 2:
            for ti in range(abs_err.shape[1]):
                for pp in [c for c in stats_all.keys() if c.startswith("p")]:
                    tgt_cols.append(f"{PROPERTY_NAMES[ti]}_{pp}")
        header = base_cols + tgt_cols

        write_header = not os.path.exists(pct_csv)
        with open(pct_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)

            row = [
                epoch + 1,
                stats_all.get("mean", np.nan), stats_all.get("max", np.nan),
            ] + [stats_all.get(k, np.nan) for k in stats_all.keys() if k.startswith("p")] + [
                stats_smax.get("mean", np.nan), stats_smax.get("max", np.nan),
            ] + [stats_smax.get(k, np.nan) for k in stats_smax.keys() if k.startswith("p")]

            if abs_err.ndim == 2:
                for ti in range(abs_err.shape[1]):
                    st = compute_error_percentiles(abs_err[:, ti], p)
                    for pp in [c for c in stats_all.keys() if c.startswith("p")]:
                        row.append(st.get(pp, np.nan))

            w.writerow(row)

    # Per-target distribution stats (helps isolate "one bad target")
    if abs_err.ndim == 2:
        for ti in range(abs_err.shape[1]):
            st = compute_error_percentiles(abs_err[:, ti], p)
            logging.info(
                f"  Val |error| stats ({PROPERTY_NAMES[ti]}): " +
                ", ".join([f"{k}={v:.6g}" for k, v in st.items()])
            )



    # Worst-k samples (for outlier inspection + stability across epochs)
    k = int(hp.get("worst_k", 10))
    every = int(hp.get("log_worst_k_every", 5))

    need_worst = (k > 0) and (
        (tracker is not None) or (every > 0 and ((epoch + 1) % every == 0))
    )
    if need_worst:
        worst_rows = get_worst_k_samples(val_true, val_pred, identifiers, k=k, reduce="max")

        # Enrich rows with which target dominates the error for that sample.
        if worst_rows:
            for r in worst_rows:
                ae = r.get("abs_err", [])
                if isinstance(ae, (list, tuple)) and len(ae) > 0:
                    t_idx = int(np.argmax(np.array(ae, dtype=float)))
                    r["worst_target_idx"] = t_idx
                    r["worst_target"] = PROPERTY_NAMES[t_idx] if t_idx < len(PROPERTY_NAMES) else f"Target_{t_idx}"
                    r["worst_abs_err"] = float(ae[t_idx])
                    r["worst_true"] = float(r["true"][t_idx])
                    r["worst_pred"] = float(r["pred"][t_idx])
                else:
                    r["worst_target_idx"] = 0
                    r["worst_target"] = PROPERTY_NAMES[0] if len(PROPERTY_NAMES) else "Target_0"
                    r["worst_abs_err"] = float(r.get("score", np.nan))
                    r["worst_true"] = float(r["true"][0]) if isinstance(r.get("true", None), (list, tuple)) and len(r["true"]) else float("nan")
                    r["worst_pred"] = float(r["pred"][0]) if isinstance(r.get("pred", None), (list, tuple)) and len(r["pred"]) else float("nan")

        # Stability check: are worst-k dominated by the same molecule/target each epoch?
        if tracker is not None:
            prev_pairs = tracker.get("prev_pairs", None)
            prev_top1 = tracker.get("prev_top1", None)

            curr_pairs = set()
            top1_pair = None
            if worst_rows:
                curr_pairs = set((str(r["id"]), int(r.get("worst_target_idx", 0))) for r in worst_rows)
                top1 = worst_rows[0]
                top1_pair = (str(top1["id"]), int(top1.get("worst_target_idx", 0)))

            if prev_pairs is not None and curr_pairs:
                overlap = len(prev_pairs & curr_pairs)
                logging.info(f"  Worst-{k} overlap vs prev epoch: {overlap}/{len(curr_pairs)} ({overlap/len(curr_pairs)*100:.1f}%)")
            if prev_top1 is not None and top1_pair is not None:
                logging.info(f"  Worst-1 same as prev epoch: {bool(prev_top1 == top1_pair)} (prev={prev_top1}, curr={top1_pair})")

            # Update tracker
            tracker["prev_pairs"] = curr_pairs
            tracker["prev_top1"] = top1_pair

            # Count persistence across epochs
            counts = tracker.setdefault("pair_counts", {})
            for p_ in curr_pairs:
                counts[p_] = int(counts.get(p_, 0)) + 1

            # Append stability log row (CSV)
            if bool(hp.get("save_epoch_reports", True)):
                os.makedirs(REPORT_DIR, exist_ok=True)
                stability_csv = os.path.join(REPORT_DIR, "val_worstk_stability.csv")
                header = ["epoch", "k", "overlap_prev", "overlap_prev_frac", "top1_same_prev", "top1_id", "top1_target_idx"]
                # compute overlap metrics for current epoch (vs previous) if available
                overlap_prev = ""
                overlap_prev_frac = ""
                top1_same = ""
                if prev_pairs is not None and curr_pairs:
                    overlap_prev = len(prev_pairs & curr_pairs)
                    overlap_prev_frac = float(overlap_prev) / float(len(curr_pairs))
                if prev_top1 is not None and top1_pair is not None:
                    top1_same = int(prev_top1 == top1_pair)
                top1_id = top1_pair[0] if top1_pair is not None else ""
                top1_ti = top1_pair[1] if top1_pair is not None else ""
                row = [epoch + 1, k, overlap_prev, overlap_prev_frac, top1_same, top1_id, top1_ti]

                import csv
                write_header = not os.path.exists(stability_csv)
                with open(stability_csv, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(header)
                    w.writerow(row)

        # Logging and CSV dumps only every 'every' epochs (to avoid huge logs)
        if k > 0 and every > 0 and ((epoch + 1) % every == 0):
            if worst_rows:
                logging.info(f"  Worst-{k} samples by max(|err|) across targets:")
                for r in worst_rows[:k]:
                    logging.info(
                        f"    id={r['id']} score={r['score']:.6f} "
                        f"worst_target={r.get('worst_target','')} worst_abs_err={r.get('worst_abs_err', float('nan')):.6f}"
                    )

            if bool(hp.get("save_epoch_reports", True)):
                os.makedirs(REPORT_DIR, exist_ok=True)
                out_csv = os.path.join(REPORT_DIR, f"val_worstk_epoch{epoch+1:03d}.csv")
                save_worst_k_csv(worst_rows, out_csv)

            # Per-target worst-k (helps isolate if only one target drives MAE)
            if val_true.ndim == 2 and val_true.shape[1] >= 1:
                for ti in range(val_true.shape[1]):
                    w_t = get_worst_k_samples(val_true[:, [ti]], val_pred[:, [ti]], identifiers, k=k, reduce="max")
                    if w_t:
                        logging.info(f"  Worst-{k} samples for target={PROPERTY_NAMES[ti]}:")
                        for r in w_t[:k]:
                            logging.info(f"    id={r['id']} err={r['abs_err'][0]:.6f} true={r['true'][0]} pred={r['pred'][0]}")
                        if bool(hp.get("save_epoch_reports", True)):
                            out_csv_t = os.path.join(REPORT_DIR, f"val_worstk_{ti}_epoch{epoch+1:03d}.csv")
                            save_worst_k_csv(w_t, out_csv_t)

    logging.info(f"  Val R2(mean)={r2_mean:.4f}, Val MAE(mean)={mae_mean:.6f}")
    return mae_mean


def train_one_run(
    data_list: List[Tuple[Any, Any]],
    num_node_features: int,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    label_median: Optional[np.ndarray],
    label_iqr: Optional[np.ndarray],
    hp: Dict[str, Any],
    device: torch.device,
    make_plots: bool = True,
    skip_test: bool = False,
) -> Dict[str, Any]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_list, train_indices, val_indices, test_indices, hp["batch_size"]
    )

    model = build_model(num_node_features, hp, device)
    criterion = build_loss(hp)
    optimizer = build_optimizer(model, hp)
    scheduler = build_scheduler(optimizer, hp)

    best_val_mae = float("inf")
    best_epoch = -1
    train_losses: List[float] = []
    val_losses: List[float] = []

    # Track worst-k stability across epochs (id + target)
    worst_tracker: Dict[str, Any] = {"prev_pairs": None, "prev_top1": None, "pair_counts": {}}

    for epoch in range(int(hp["epochs"])):
        train_out = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip_norm=hp.get("grad_clip_norm", None),
            log_grad_norm=False,
        )
        train_loss = float(train_out['loss']) if isinstance(train_out, dict) else float(train_out[0])
        grad_stats = None
        if isinstance(train_out, dict):
            if 'grad_norm_mean' in train_out or 'grad_norm_max' in train_out:
                grad_stats = {k: train_out.get(k) for k in ('grad_norm_mean','grad_norm_max') if k in train_out}
        elif isinstance(train_out, (list, tuple)) and len(train_out) > 1:
            grad_stats = train_out[1]
        val_loss, val_pred_scaled, val_true_scaled, val_ids = evaluate_epoch(model, val_loader, criterion, device)

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

        # Inverse transform labels for MAE/R2 reporting.
        val_pred = val_pred_scaled
        val_true = val_true_scaled
        if label_median is not None and label_iqr is not None and val_pred.size and val_true.size:
            val_pred = inverse_scale_labels(val_pred_scaled, label_median, label_iqr)
            val_true = inverse_scale_labels(val_true_scaled, label_median, label_iqr)

        logging.info(
            f"Epoch {epoch+1}/{hp['epochs']}: "
            f"TrainLoss={train_loss:.6f}, ValLoss={val_loss:.6f}"
        )

        if val_pred.size == 0 or val_true.size == 0:
            continue

        val_mae = log_val_diagnostics(epoch, hp, val_true, val_pred, val_ids, tracker=worst_tracker)

        # Scheduler step (Plateau)
        if scheduler is not None:
            metric = str(hp.get("scheduler_metric", "mae")).lower()
            to_monitor = float(val_mae) if metric == "mae" else float(val_loss)
            scheduler.step(to_monitor)

        # Checkpoint on MAE (goal metric)
        if val_mae < best_val_mae:
            best_val_mae = float(val_mae)
            best_epoch = int(epoch + 1)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logging.info(f"  Saved best model (val MAE={best_val_mae:.6f}) at epoch {best_epoch}")

    result: Dict[str, Any] = {"best_val_mae": best_val_mae, "best_epoch": best_epoch}


    # Select final checkpoint path (stage-1 best by default)
    final_model_path = MODEL_SAVE_PATH
    final_best_val_mae = float(best_val_mae)
    final_best_epoch = int(best_epoch)

    # -------------------------
    # Stage 2: hard-example mining fine-tune (optional)
    # -------------------------
    if bool(hp.get("enable_hard_mining", False)):
        # Load stage-1 best weights
        model.load_state_dict(torch.load(final_model_path, map_location=device))
        model.eval()

        # Evaluate TRAIN set (no shuffle) to identify hard examples.
        train_eval_loader = DataLoader([data_list[i] for i in train_indices], batch_size=hp["batch_size"], shuffle=False)
        _, tr_pred_s, tr_true_s, tr_ids = evaluate_epoch(model, train_eval_loader, criterion, device)

        if tr_pred_s.size and tr_true_s.size:
            tr_pred = tr_pred_s
            tr_true = tr_true_s
            if label_median is not None and label_iqr is not None:
                tr_pred = inverse_scale_labels(tr_pred_s, label_median, label_iqr)
                tr_true = inverse_scale_labels(tr_true_s, label_median, label_iqr)

            # Hardness score per-sample: max(|err|) across targets.
            tr_abs_err = np.abs(tr_pred - tr_true)
            if tr_abs_err.ndim == 1:
                tr_abs_err = tr_abs_err.reshape(-1, 1)
            tr_score = tr_abs_err.max(axis=1)

            top_k = int(hp.get("hard_mining_top_k", 0) or 0)
            if top_k <= 0:
                pct = float(hp.get("hard_mining_percent", 0.0) or 0.0)
                top_k = max(1, int(round(pct * len(train_indices)))) if pct > 0 else 0
            top_k = min(max(top_k, 0), len(train_indices))

            if top_k > 0:
                hard_pos = np.argsort(-tr_score)[:top_k].tolist()

                # Save hard-mining report (train).
                if bool(hp.get("save_epoch_reports", True)):
                    os.makedirs(REPORT_DIR, exist_ok=True)
                    import csv
                    out_csv = os.path.join(REPORT_DIR, "train_hard_mining_topk.csv")
                    header = ["rank", "id", "dataset_index", "score_max_abs_err", "true", "pred", "signed_err"]
                    with open(out_csv, "w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow(header)
                        for rnk, p in enumerate(hard_pos, start=1):
                            true_v = float(tr_true[p, 0]) if tr_true.ndim == 2 else float(tr_true[p])
                            pred_v = float(tr_pred[p, 0]) if tr_pred.ndim == 2 else float(tr_pred[p])
                            w.writerow([rnk, str(tr_ids[p]), int(train_indices[p]), float(tr_score[p]), true_v, pred_v, pred_v - true_v])

                # Build oversampled stage-2 training indices (duplicates allowed).
                mult = int(hp.get("hard_oversample_mult", 1) or 1)
                mult = max(1, mult)
                stage2_train_indices = list(train_indices)
                if mult > 1:
                    for p in hard_pos:
                        stage2_train_indices.extend([train_indices[p]] * (mult - 1))
                random.shuffle(stage2_train_indices)

                stage2_train_loader = DataLoader(
                    [data_list[i] for i in stage2_train_indices],
                    batch_size=hp["batch_size"],
                    shuffle=True,
                )

                # Stage-2 hyperparameters (small LR fine-tune, optionally different loss/scheduler)
                stage2_hp = dict(hp)
                stage2_hp["epochs"] = int(hp.get("stage2_epochs", 0) or 0)
                stage2_hp["learning_rate"] = float(hp.get("stage2_learning_rate", hp.get("learning_rate", 1e-3)))
                stage2_hp["optimizer"] = str(hp.get("stage2_optimizer", hp.get("optimizer", "adamw")))
                stage2_hp["weight_decay"] = float(hp.get("stage2_weight_decay", hp.get("weight_decay", 0.0)))
                stage2_hp["loss_type"] = str(hp.get("stage2_loss_type", hp.get("loss_type", "huber")))
                stage2_hp["huber_delta"] = float(hp.get("stage2_huber_delta", hp.get("huber_delta", 1.0)))
                stage2_hp["scheduler"] = str(hp.get("stage2_scheduler", hp.get("scheduler", "none")))
                stage2_hp["scheduler_metric"] = str(hp.get("stage2_scheduler_metric", hp.get("scheduler_metric", "mae")))
                stage2_hp["scheduler_patience"] = int(hp.get("stage2_scheduler_patience", hp.get("scheduler_patience", 5)))
                stage2_hp["scheduler_factor"] = float(hp.get("stage2_scheduler_factor", hp.get("scheduler_factor", 0.5)))
                stage2_hp["scheduler_min_lr"] = float(hp.get("stage2_scheduler_min_lr", hp.get("scheduler_min_lr", 1e-6)))

                stage2_epochs = int(stage2_hp["epochs"])
                if stage2_epochs > 0:
                    logging.info(
                        f"Stage-2 hard-mining fine-tune: top_k={top_k}, mult={mult}, "
                        f"epochs={stage2_epochs}, lr={stage2_hp['learning_rate']}, loss={stage2_hp['loss_type']}"
                    )

                    stage2_criterion = build_loss(stage2_hp)
                    stage2_optimizer = build_optimizer(model, stage2_hp)
                    stage2_scheduler = build_scheduler(stage2_optimizer, stage2_hp)

                    stage2_best_val_mae = float("inf")
                    stage2_best_epoch = -1

                    # Separate tracker for stage-2 to avoid mixing epoch counts.
                    stage2_worst_tracker: Dict[str, Any] = {"prev_pairs": None, "prev_top1": None, "pair_counts": {}}

                    for e2 in range(stage2_epochs):
                        train2_out = train_epoch(
                            model,
                            stage2_train_loader,
                            stage2_criterion,
                            stage2_optimizer,
                            device,
                            grad_clip_norm=stage2_hp.get("grad_clip_norm", None),
                            log_grad_norm=False,
                        )
                        tr_loss2 = float(train2_out['loss']) if isinstance(train2_out, dict) else float(train2_out[0])
                        v_loss2, vpred_s2, vtrue_s2, vids2 = evaluate_epoch(model, val_loader, stage2_criterion, device)

                        vpred2 = vpred_s2
                        vtrue2 = vtrue_s2
                        if label_median is not None and label_iqr is not None and vpred2.size and vtrue2.size:
                            vpred2 = inverse_scale_labels(vpred_s2, label_median, label_iqr)
                            vtrue2 = inverse_scale_labels(vtrue_s2, label_median, label_iqr)

                        logging.info(
                            f"Stage2 Epoch {e2+1}/{stage2_epochs}: TrainLoss={tr_loss2:.6f}, ValLoss={v_loss2:.6f}"
                        )
                        if vpred2.size and vtrue2.size:
                            v_mae2 = log_val_diagnostics(
                                e2, stage2_hp, vtrue2, vpred2, vids2, tracker=stage2_worst_tracker
                            )
                        else:
                            v_mae2 = float("inf")

                        if stage2_scheduler is not None:
                            metric = str(stage2_hp.get("scheduler_metric", "mae")).lower()
                            to_monitor = float(v_mae2) if metric == "mae" else float(v_loss2)
                            stage2_scheduler.step(to_monitor)

                        if v_mae2 < stage2_best_val_mae:
                            stage2_best_val_mae = float(v_mae2)
                            stage2_best_epoch = int(e2 + 1)
                            torch.save(model.state_dict(), STAGE2_MODEL_SAVE_PATH)
                            logging.info(
                                f"  Saved best stage-2 model (val MAE={stage2_best_val_mae:.6f}) at epoch {stage2_best_epoch}"
                            )

                    # Select final model path based on best val MAE among stage-1 and stage-2.
                    if stage2_best_val_mae < final_best_val_mae:
                        final_model_path = STAGE2_MODEL_SAVE_PATH
                        final_best_val_mae = float(stage2_best_val_mae)
                        final_best_epoch = int(stage2_best_epoch)
                        result["best_val_mae_stage2"] = float(stage2_best_val_mae)
                        result["best_epoch_stage2"] = int(stage2_best_epoch)
            else:
                logging.info("Hard-mining: top_k==0; stage-2 skipped.")
        else:
            logging.info("Hard-mining: train predictions unavailable; stage-2 skipped.")


    # Test evaluation with best checkpoint
    if skip_test:
        return result

    if len(test_loader.dataset) == 0:
        logging.warning("Test dataset is empty.")
        return result

    # Load the best checkpoint for final reporting/testing.
    # If stage-2 was enabled and produced a checkpoint, prefer it; otherwise fall back to stage-1 best.
    final_ckpt_path = MODEL_SAVE_PATH
    if bool(hp.get("enable_hard_mining", False)) and os.path.exists(STAGE2_MODEL_SAVE_PATH):
        final_ckpt_path = STAGE2_MODEL_SAVE_PATH

    try:
        state = torch.load(final_ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        # Older PyTorch versions do not support weights_only
        state = torch.load(final_ckpt_path, map_location=device)
    model.load_state_dict(state)

    # For final reporting (best checkpoint), build an id->SMILES map once.
    id2smiles = _load_id_to_smiles_map() if bool(hp.get("include_smiles_in_worstk", True)) else {}
    final_k = int(hp.get("final_worst_k", hp.get("worst_k", 20)))


    # Save worst-k on train (best checkpoint) to see whether the same molecules are also hard in TRAIN.
    if final_k > 0 and bool(hp.get("save_train_worstk_best_checkpoint", True)):
        train_eval_loader2 = DataLoader([data_list[i] for i in train_indices], batch_size=hp["batch_size"], shuffle=False)
        _, tpred_s, ttrue_s, tids = evaluate_epoch(model, train_eval_loader2, criterion, device)
        if tpred_s.size and ttrue_s.size:
            tpred = tpred_s
            ttrue = ttrue_s
            if label_median is not None and label_iqr is not None:
                tpred = inverse_scale_labels(tpred_s, label_median, label_iqr)
                ttrue = inverse_scale_labels(ttrue_s, label_median, label_iqr)
            worst_train = get_worst_k_samples(ttrue, tpred, tids, k=final_k, reduce="max")
            for r in worst_train:
                r["split"] = "train"
            _enrich_worst_rows_with_smiles(worst_train, id2smiles)
            out_csv = os.path.join(REPORT_DIR, "train_worstk_best_checkpoint.csv")
            save_worst_k_csv(worst_train, out_csv)


    # Save worst-k on validation (best checkpoint) to help isolate persistent outliers.
    if final_k > 0 and len(val_loader.dataset) > 0 and bool(hp.get("save_final_worst_k", True)):
        _, vpred_s, vtrue_s, vids = evaluate_epoch(model, val_loader, criterion, device)
        if vpred_s.size and vtrue_s.size:
            vpred = vpred_s
            vtrue = vtrue_s
            if label_median is not None and label_iqr is not None:
                vpred = inverse_scale_labels(vpred_s, label_median, label_iqr)
                vtrue = inverse_scale_labels(vtrue_s, label_median, label_iqr)

            worst_val = get_worst_k_samples(vtrue, vpred, vids, k=final_k, reduce="max")
            for r in worst_val:
                r["split"] = "val"
            _enrich_worst_rows_with_smiles(worst_val, id2smiles)
            out_csv = os.path.join(REPORT_DIR, "val_worstk_best_checkpoint.csv")
            save_worst_k_csv(worst_val, out_csv)
    _, test_pred_scaled, test_true_scaled, test_ids = evaluate_epoch(model, test_loader, criterion, device)

    if test_pred_scaled.size == 0 or test_true_scaled.size == 0:
        logging.warning("No predictions/targets on test set.")
        return result

    test_pred = test_pred_scaled
    test_true = test_true_scaled
    if label_median is not None and label_iqr is not None:
        test_pred = inverse_scale_labels(test_pred_scaled, label_median, label_iqr)
        test_true = inverse_scale_labels(test_true_scaled, label_median, label_iqr)

    final_metrics = evaluate_metrics(test_pred, test_true, PROPERTY_NAMES)
    result["test_metrics"] = final_metrics

    # Save worst-k on test (best checkpoint).
    if final_k > 0 and bool(hp.get("save_final_worst_k", True)):
        worst_test = get_worst_k_samples(test_true, test_pred, test_ids, k=final_k, reduce="max")
        for r in worst_test:
            r["split"] = "test"
        _enrich_worst_rows_with_smiles(worst_test, id2smiles)
        out_csv = os.path.join(REPORT_DIR, "test_worstk_best_checkpoint.csv")
        save_worst_k_csv(worst_test, out_csv)

    if make_plots:
        plot_results(
            train_losses,
            val_losses,
            test_true,
            test_pred,
            PROPERTY_NAMES,
            metrics=final_metrics,
            output_dir=OUTPUT_DIR,
        )

    return result


def run_ablation(
    data_list: List[Tuple[Any, Any]],
    num_node_features: int,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    label_median: Optional[np.ndarray],
    label_iqr: Optional[np.ndarray],
    base_hp: Dict[str, Any],
    device: torch.device,
) -> None:
    """Small grid: huber_delta × grad_clip_norm × scheduler.

    Produces a CSV table (reproducible, fixed split + fixed seed policy).
    """
    deltas = [float(x) for x in base_hp.get("ablation_huber_deltas", [0.5, 1.0, 2.0])]
    clips = [float(x) for x in base_hp.get("ablation_grad_clips", [0.0, 1.0, 2.0])]
    scheds = [str(x) for x in base_hp.get("ablation_schedulers", ["none", "plateau"])]
    repeats = int(base_hp.get("ablation_repeats", 1))
    ablation_epochs = int(base_hp.get("ablation_epochs", 30))

    rows: List[Dict[str, Any]] = []
    run_id = 0
    for delta in deltas:
        for clip in clips:
            for sched in scheds:
                for rep in range(repeats):
                    run_id += 1
                    hp = dict(base_hp)
                    hp["epochs"] = ablation_epochs
                    hp["loss_type"] = "huber"
                    hp["huber_delta"] = float(delta)
                    hp["grad_clip_norm"] = float(clip) if clip > 0 else None
                    hp["scheduler"] = str(sched).lower()
                    hp["save_epoch_reports"] = False  # keep ablation light
                    hp["log_worst_k_every"] = 0

                    # Seed policy: deterministic but unique per run
                    set_seed(SEED + run_id)

                    logging.info(f"[Ablation {run_id}] delta={delta} clip={clip} sched={sched} rep={rep+1}/{repeats}")
                    res = train_one_run(
                        data_list,
                        num_node_features,
                        train_indices,
                        val_indices,
                        test_indices,
                        label_median,
                        label_iqr,
                        hp,
                        device,
                        make_plots=False,
                        skip_test=True,
                    )
                    rows.append({
                        "run_id": run_id,
                        "huber_delta": delta,
                        "grad_clip_norm": clip,
                        "scheduler": sched,
                        "best_val_mae": res.get("best_val_mae", np.nan),
                        "best_epoch": res.get("best_epoch", -1),
                    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import csv
    with open(ABLATION_RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            for r in rows:
                w.writerow(r)
    logging.info(f"Ablation results saved: {ABLATION_RESULTS_PATH}")

    if rows:
        rows_sorted = sorted(rows, key=lambda r: float(r["best_val_mae"]))
        top = rows_sorted[: min(10, len(rows_sorted))]
        logging.info("Top ablation configs (by best_val_mae):")
        for r in top:
            logging.info(
                f"  run={r['run_id']} mae={float(r['best_val_mae']):.6f} "
                f"delta={r['huber_delta']} clip={r['grad_clip_norm']} sched={r['scheduler']} "
                f"best_epoch={r['best_epoch']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true", help="run small hyperparameter grid and write ablation table")
    parser.add_argument("--no-plots", action="store_true", help="disable plots")
    args = parser.parse_args()

    logging.basicConfig(level=LOGGING_LEVEL, format=LOG_FORMAT)
    set_seed(SEED)

    data_list, num_node_features = load_and_preprocess_qm9_data()
    if not data_list:
        logging.error("No valid data loaded. Exiting.")
        return

    indices = list(range(len(data_list)))

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=float(HYPERPARAMS["test_split_ratio"]),
        random_state=int(HYPERPARAMS["random_state"]),
    )
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        random_state=int(HYPERPARAMS["random_state"]),
    )
    logging.info(f"Data split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")

    # Label scaling (robust median/IQR) fit on TRAIN only
    label_median, label_iqr = calculate_label_scaling_params(data_list, train_indices)
    if label_median is not None and label_iqr is not None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.savez(LABEL_SCALING_PARAMS_PATH, median=label_median, iqr=label_iqr)
        apply_label_scaling(data_list, label_median, label_iqr)

    # Global descriptor scaling: sklearn StandardScaler fit on TRAIN only
    g_mean, g_std = fit_global_standard_scaler(data_list, train_indices)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(GLOBAL_SCALING_PARAMS_PATH, mean=g_mean, std=g_std)
    clip_value = float(HYPERPARAMS.get("global_clip_value", 0.0))
    apply_global_scaling(data_list, g_mean, g_std, clip_value=clip_value if clip_value > 0 else None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if args.ablation or bool(HYPERPARAMS.get("run_ablation", False)):
        run_ablation(
            data_list,
            num_node_features,
            train_indices,
            val_indices,
            test_indices,
            label_median,
            label_iqr,
            HYPERPARAMS,
            device,
        )
        return

    train_one_run(
        data_list,
        num_node_features,
        train_indices,
        val_indices,
        test_indices,
        label_median,
        label_iqr,
        HYPERPARAMS,
        device,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()