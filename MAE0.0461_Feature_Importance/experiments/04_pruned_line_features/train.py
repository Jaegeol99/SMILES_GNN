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
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader

from config import (
    CSV_INDEX_COL,
    CSV_PATH,
    CSV_SMILES_COL,
    FEATURE_IMPORTANCE_CSV_PATH,
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
    HYB_DIM,
    LAPLACIAN_PE_K,
    LINE_NODE_FEATURE_DIM,
    NUM_BOND_FEATURES,
    NUM_LINE_EDGE_FEATURES,
    RING_STRAIN_DIM,
    TOTAL_FEATURE_DIMENSION,
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_global_standard_scaler(
    data_list: List[Any],
    train_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] =[]
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
        edge_in_dim=NUM_BOND_FEATURES,
        line_node_in_dim=LINE_NODE_FEATURE_DIM,
        line_edge_in_dim=NUM_LINE_EDGE_FEATURES,
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


def build_noam_scheduler(optimizer, epochs, steps_per_epoch):
    warmup_epochs = 2.0
    init_lr = 1e-4
    max_lr = 1e-3
    final_lr = 1e-4
    
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    decay_steps = int((epochs - warmup_epochs) * steps_per_epoch)
    
    def lr_lambda(step):
        if step < warmup_steps:
            lr = init_lr + (max_lr - init_lr) * step / max(1, warmup_steps)
        else:
            if decay_steps <= 0:
                lr = final_lr
            else:
                factor = (final_lr / max_lr) ** (1.0 / decay_steps)
                lr = max_lr * (factor ** (step - warmup_steps))
        # LambdaLR expects a multiplier relative to the optimizer base LR.
        return lr / optimizer.defaults.get("lr", 1e-3)
            
    return LambdaLR(optimizer, lr_lambda)


def _make_block_permutation_specs() -> List[Dict[str, Any]]:
    line_node_bond_start = TOTAL_FEATURE_DIMENSION

    line_edge_in_bond_start = HYB_DIM
    line_edge_out_bond_start = line_edge_in_bond_start + NUM_BOND_FEATURES
    line_edge_ring_strain_start = line_edge_out_bond_start + NUM_BOND_FEATURES

    return [
        {
            "name": "atom_node_features",
            "tensor_group": "atom.x",
            "targets": [("atom", "x", 0, TOTAL_FEATURE_DIMENSION)],
            "block_size": TOTAL_FEATURE_DIMENSION,
        },
        {
            "name": "line_src_atom_features",
            "tensor_group": "line.x",
            "targets": [("line", "x", 0, TOTAL_FEATURE_DIMENSION)],
            "block_size": TOTAL_FEATURE_DIMENSION,
        },
        {
            "name": "line_bond_features",
            "tensor_group": "line.x",
            "targets": [("line", "x", line_node_bond_start, line_node_bond_start + NUM_BOND_FEATURES)],
            "block_size": NUM_BOND_FEATURES,
        },
        {
            "name": "line_center_hybridization",
            "tensor_group": "line.edge_attr",
            "targets": [("line", "edge_attr", 0, HYB_DIM)],
            "block_size": HYB_DIM,
        },
        {
            "name": "line_incoming_bond_features",
            "tensor_group": "line.edge_attr",
            "targets": [("line", "edge_attr", line_edge_in_bond_start, line_edge_in_bond_start + NUM_BOND_FEATURES)],
            "block_size": NUM_BOND_FEATURES,
        },
        {
            "name": "line_outgoing_bond_features",
            "tensor_group": "line.edge_attr",
            "targets": [("line", "edge_attr", line_edge_out_bond_start, line_edge_out_bond_start + NUM_BOND_FEATURES)],
            "block_size": NUM_BOND_FEATURES,
        },
        {
            "name": "line_ring_strain",
            "tensor_group": "line.edge_attr",
            "targets": [("line", "edge_attr", line_edge_ring_strain_start, line_edge_ring_strain_start + RING_STRAIN_DIM)],
            "block_size": RING_STRAIN_DIM,
        },
        {
            "name": "global_descriptors",
            "tensor_group": "atom.g",
            "targets": [("atom", "g", 0, GLOBAL_FEATURE_DIM)],
            "block_size": GLOBAL_FEATURE_DIM,
        },
    ]


def _permute_tensor_block_rows_(tensor: Optional[torch.Tensor], start: int, end: int, rng: np.random.Generator) -> None:
    if tensor is None or tensor.numel() == 0 or tensor.dim() != 2:
        return
    n_rows, n_cols = int(tensor.size(0)), int(tensor.size(1))
    if n_rows < 2 or start >= n_cols:
        return
    end = min(int(end), n_cols)
    if end <= start:
        return

    perm = torch.as_tensor(rng.permutation(n_rows), device=tensor.device, dtype=torch.long)
    permuted = tensor.index_select(0, perm)
    tensor[:, start:end] = permuted[:, start:end]


def _apply_block_permutation_(
    atom_batch: Any,
    line_batch: Any,
    block_spec: Dict[str, Any],
    rng: np.random.Generator,
) -> None:
    for batch_name, attr_name, start, end in block_spec.get("targets", []):
        batch_obj = atom_batch if batch_name == "atom" else line_batch
        tensor = getattr(batch_obj, attr_name, None)
        _permute_tensor_block_rows_(tensor, int(start), int(end), rng)


def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    block_spec: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    was_training = model.training
    model.eval()

    predictions_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if not (isinstance(batch, (list, tuple)) and len(batch) == 2):
                logging.warning("Invalid batch during feature importance evaluation, skipping.")
                continue

            atom_batch, line_batch = batch
            atom_batch = atom_batch.to(device)
            line_batch = line_batch.to(device)

            if block_spec is not None:
                rng = np.random.default_rng(int(seed) + int(batch_idx))
                _apply_block_permutation_(atom_batch, line_batch, block_spec, rng)

            output, _ = model(atom_batch, line_batch)
            target = atom_batch.y.view_as(output)

            predictions_list.append(output.detach().cpu().numpy())
            targets_list.append(target.detach().cpu().numpy())

    if was_training:
        model.train()

    predictions = np.concatenate(predictions_list, axis=0) if predictions_list else np.array([])
    targets = np.concatenate(targets_list, axis=0) if targets_list else np.array([])
    return predictions, targets


def _inverse_scale_if_needed(
    values: np.ndarray,
    label_median: Optional[np.ndarray],
    label_iqr: Optional[np.ndarray],
) -> np.ndarray:
    if values.size == 0 or label_median is None or label_iqr is None:
        return values
    return inverse_scale_labels(values, label_median, label_iqr)


def _compute_mean_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0 or y_pred.size == 0 or y_true.shape != y_pred.shape:
        return float("nan")
    return float(mean_absolute_error(y_true, y_pred))


def _resolve_feature_importance_loader(
    data_list: List[Any],
    train_indices: List[int],
    val_loader: DataLoader,
    test_loader: DataLoader,
    hp: Dict[str, Any],
    skip_test: bool,
) -> Tuple[Optional[str], Optional[DataLoader]]:
    requested = str(hp.get("feature_importance_split", "test")).strip().lower()
    if requested not in {"train", "val", "test"}:
        requested = "test"

    candidates = [requested] + [split for split in ("test", "val", "train") if split != requested]
    if skip_test:
        candidates = [split for split in candidates if split != "test"]

    for split in candidates:
        if split == "test" and len(test_loader.dataset) > 0:
            return split, test_loader
        if split == "val" and len(val_loader.dataset) > 0:
            return split, val_loader
        if split == "train" and len(train_indices) > 0:
            return split, DataLoader(
                [data_list[i] for i in train_indices],
                batch_size=hp["batch_size"],
                shuffle=False,
            )
    return None, None


def compute_block_permutation_importance(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_median: Optional[np.ndarray],
    label_iqr: Optional[np.ndarray],
    repeats: int,
    seed: int,
    split_name: str,
) -> List[Dict[str, Any]]:
    baseline_pred_s, baseline_true_s = _collect_predictions(model, loader, device, block_spec=None, seed=seed)
    if baseline_pred_s.size == 0 or baseline_true_s.size == 0:
        return []

    baseline_pred = _inverse_scale_if_needed(baseline_pred_s, label_median, label_iqr)
    baseline_true = _inverse_scale_if_needed(baseline_true_s, label_median, label_iqr)
    baseline_mae = _compute_mean_mae(baseline_true, baseline_pred)

    rows: List[Dict[str, Any]] = []
    block_specs = _make_block_permutation_specs()
    repeats = max(int(repeats), 1)

    for block_idx, block_spec in enumerate(block_specs):
        logging.info(
            f"Feature importance [{block_idx + 1}/{len(block_specs)}]: "
            f"permuting {block_spec['name']} on {split_name} split"
        )
        permuted_maes: List[float] = []

        for repeat_idx in range(repeats):
            repeat_seed = int(seed) + (block_idx * 1000) + repeat_idx
            perm_pred_s, perm_true_s = _collect_predictions(
                model,
                loader,
                device,
                block_spec=block_spec,
                seed=repeat_seed,
            )
            perm_pred = _inverse_scale_if_needed(perm_pred_s, label_median, label_iqr)
            perm_true = _inverse_scale_if_needed(perm_true_s, label_median, label_iqr)
            permuted_maes.append(_compute_mean_mae(perm_true, perm_pred))

        permuted_mae_mean = float(np.mean(permuted_maes))
        permuted_mae_std = float(np.std(permuted_maes))
        importance_delta = float(permuted_mae_mean - baseline_mae)
        relative_increase_pct = float((importance_delta / max(abs(baseline_mae), 1e-12)) * 100.0)

        rows.append(
            {
                "split": split_name,
                "block_name": block_spec["name"],
                "tensor_group": block_spec["tensor_group"],
                "block_size": int(block_spec["block_size"]),
                "repeats": repeats,
                "baseline_mae": float(baseline_mae),
                "permuted_mae_mean": permuted_mae_mean,
                "permuted_mae_std": permuted_mae_std,
                "importance_mae_delta": importance_delta,
                "relative_increase_pct": relative_increase_pct,
            }
        )

    rows.sort(key=lambda row: row["importance_mae_delta"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def save_block_permutation_importance_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        return

    import csv

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["rank"] + [key for key in rows[0].keys() if key != "rank"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def log_val_diagnostics(
    epoch: int,
    hp: Dict[str, Any],
    val_true: np.ndarray,
    val_pred: np.ndarray,
    identifiers: List[str],
    tracker: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    r2_per = r2_score(val_true, val_pred, multioutput="raw_values")
    mae_per = mean_absolute_error(val_true, val_pred, multioutput="raw_values")
    r2_mean = float(np.mean(r2_per))
    mae_mean = float(np.mean(mae_per))

    abs_err = np.abs(val_pred - val_true)
    p =[int(x) for x in hp.get("error_percentiles", [50, 90, 95, 99])]
    stats_all = compute_error_percentiles(abs_err, p)

    if abs_err.ndim == 2:
        max_per_sample = abs_err.max(axis=1)
    else:
        max_per_sample = abs_err
    stats_smax = compute_error_percentiles(max_per_sample, p)

    if bool(hp.get("save_epoch_reports", True)):
        os.makedirs(REPORT_DIR, exist_ok=True)
        pct_csv = os.path.join(REPORT_DIR, "val_error_percentiles.csv")
        import csv

        base_cols =[
            "epoch",
            "pooled_mean", "pooled_max",
        ] +[f"pooled_{k}" for k in stats_all.keys() if k.startswith("p")] + [
            "smax_mean", "smax_max",
        ] +[f"smax_{k}" for k in stats_smax.keys() if k.startswith("p")]
        tgt_cols: List[str] =[]
        if abs_err.ndim == 2:
            for ti in range(abs_err.shape[1]):
                for pp in[c for c in stats_all.keys() if c.startswith("p")]:
                    tgt_cols.append(f"{PROPERTY_NAMES[ti]}_{pp}")
        header = base_cols + tgt_cols

        write_header = not os.path.exists(pct_csv)
        with open(pct_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)

            row =[
                epoch + 1,
                stats_all.get("mean", np.nan), stats_all.get("max", np.nan),
            ] +[stats_all.get(k, np.nan) for k in stats_all.keys() if k.startswith("p")] +[
                stats_smax.get("mean", np.nan), stats_smax.get("max", np.nan),
            ] +[stats_smax.get(k, np.nan) for k in stats_smax.keys() if k.startswith("p")]

            if abs_err.ndim == 2:
                for ti in range(abs_err.shape[1]):
                    st = compute_error_percentiles(abs_err[:, ti], p)
                    for pp in[c for c in stats_all.keys() if c.startswith("p")]:
                        row.append(st.get(pp, np.nan))

            w.writerow(row)

    k = int(hp.get("worst_k", 10))
    every = int(hp.get("log_worst_k_every", 5))

    need_worst = (k > 0) and (
        (tracker is not None) or (every > 0 and ((epoch + 1) % every == 0))
    )
    if need_worst:
        worst_rows = get_worst_k_samples(val_true, val_pred, identifiers, k=k, reduce="max")

        if worst_rows:
            for r in worst_rows:
                ae = r.get("abs_err",[])
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
                    r["worst_true"] = float(r["true"]) if isinstance(r.get("true", None), (list, tuple)) and len(r["true"]) else float("nan")
                    r["worst_pred"] = float(r["pred"]) if isinstance(r.get("pred", None), (list, tuple)) and len(r["pred"]) else float("nan")

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
            if prev_top1 is not None and top1_pair is not None:
                _ = bool(prev_top1 == top1_pair)

            tracker["prev_pairs"] = curr_pairs
            tracker["prev_top1"] = top1_pair

            counts = tracker.setdefault("pair_counts", {})
            for p_ in curr_pairs:
                counts[p_] = int(counts.get(p_, 0)) + 1

            if bool(hp.get("save_epoch_reports", True)):
                os.makedirs(REPORT_DIR, exist_ok=True)
                stability_csv = os.path.join(REPORT_DIR, "val_worstk_stability.csv")
                header =["epoch", "k", "overlap_prev", "overlap_prev_frac", "top1_same_prev", "top1_id", "top1_target_idx"]
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
                row =[epoch + 1, k, overlap_prev, overlap_prev_frac, top1_same, top1_id, top1_ti]

                import csv
                write_header = not os.path.exists(stability_csv)
                with open(stability_csv, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(header)
                    w.writerow(row)

        if k > 0 and every > 0 and ((epoch + 1) % every == 0):
            if bool(hp.get("save_epoch_reports", True)):
                os.makedirs(REPORT_DIR, exist_ok=True)
                out_csv = os.path.join(REPORT_DIR, f"val_worstk_epoch{epoch+1:03d}.csv")
                save_worst_k_csv(worst_rows, out_csv)

            if val_true.ndim == 2 and val_true.shape[1] >= 1:
                for ti in range(val_true.shape[1]):
                    w_t = get_worst_k_samples(val_true[:, [ti]], val_pred[:, [ti]], identifiers, k=k, reduce="max")
                    if w_t and bool(hp.get("save_epoch_reports", True)):
                        out_csv_t = os.path.join(REPORT_DIR, f"val_worstk_{ti}_epoch{epoch+1:03d}.csv")
                        save_worst_k_csv(w_t, out_csv_t)

    return mae_mean, r2_mean


def train_one_run(
    data_list: List[Any],
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
    
    steps_per_epoch = len(train_loader)
    scheduler = build_noam_scheduler(optimizer, int(hp["epochs"]), steps_per_epoch)

    best_val_mae = float("inf")
    best_epoch = -1
    train_losses: List[float] = []
    val_losses: List[float] =[]

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
            scheduler=scheduler,
        )
        train_loss = float(train_out['loss']) if isinstance(train_out, dict) else float(train_out)
        
        val_loss, val_pred_scaled, val_true_scaled, val_ids = evaluate_epoch(model, val_loader, criterion, device)

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

        val_pred = val_pred_scaled
        val_true = val_true_scaled
        if label_median is not None and label_iqr is not None and val_pred.size and val_true.size:
            val_pred = inverse_scale_labels(val_pred_scaled, label_median, label_iqr)
            val_true = inverse_scale_labels(val_true_scaled, label_median, label_iqr)

        if val_pred.size == 0 or val_true.size == 0:
            logging.info(
                f"Epoch {epoch+1}/{hp['epochs']}: "
                f"TrainLoss={train_loss:.6f}, ValLoss={val_loss:.6f}, ValR2=nan, ValMAE=nan"
            )
            continue

        val_mae, val_r2 = log_val_diagnostics(epoch, hp, val_true, val_pred, val_ids, tracker=worst_tracker)
        logging.info(
            f"Epoch {epoch+1}/{hp['epochs']}: "
            f"TrainLoss={train_loss:.6f}, ValLoss={val_loss:.6f}, ValR2={val_r2:.4f}, ValMAE={val_mae:.6f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = float(val_mae)
            best_epoch = int(epoch + 1)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    result: Dict[str, Any] = {"best_val_mae": best_val_mae, "best_epoch": best_epoch}

    final_model_path = MODEL_SAVE_PATH
    final_best_val_mae = float(best_val_mae)
    final_best_epoch = int(best_epoch)
    # Optional hard-example mining fine-tune.
    if bool(hp.get("enable_hard_mining", False)):
        model.load_state_dict(torch.load(final_model_path, map_location=device))
        model.eval()

        train_eval_loader = DataLoader([data_list[i] for i in train_indices], batch_size=hp["batch_size"], shuffle=False)
        _, tr_pred_s, tr_true_s, tr_ids = evaluate_epoch(model, train_eval_loader, criterion, device)

        if tr_pred_s.size and tr_true_s.size:
            tr_pred = tr_pred_s
            tr_true = tr_true_s
            if label_median is not None and label_iqr is not None:
                tr_pred = inverse_scale_labels(tr_pred_s, label_median, label_iqr)
                tr_true = inverse_scale_labels(tr_true_s, label_median, label_iqr)

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

                if bool(hp.get("save_epoch_reports", True)):
                    os.makedirs(REPORT_DIR, exist_ok=True)
                    import csv
                    out_csv = os.path.join(REPORT_DIR, "train_hard_mining_topk.csv")
                    header =["rank", "id", "dataset_index", "score_max_abs_err", "true", "pred", "signed_err"]
                    with open(out_csv, "w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow(header)
                        for rnk, p in enumerate(hard_pos, start=1):
                            true_v = float(tr_true[p, 0]) if tr_true.ndim == 2 else float(tr_true[p])
                            pred_v = float(tr_pred[p, 0]) if tr_pred.ndim == 2 else float(tr_pred[p])
                            w.writerow([rnk, str(tr_ids[p]), int(train_indices[p]), float(tr_score[p]), true_v, pred_v, pred_v - true_v])

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

                stage2_hp = dict(hp)
                stage2_hp["epochs"] = int(hp.get("stage2_epochs", 0) or 0)
                stage2_hp["learning_rate"] = float(hp.get("stage2_learning_rate", hp.get("learning_rate", 1e-3)))
                stage2_hp["optimizer"] = str(hp.get("stage2_optimizer", hp.get("optimizer", "adamw")))
                stage2_hp["weight_decay"] = float(hp.get("stage2_weight_decay", hp.get("weight_decay", 0.0)))
                stage2_hp["loss_type"] = str(hp.get("stage2_loss_type", hp.get("loss_type", "huber")))
                stage2_hp["huber_delta"] = float(hp.get("stage2_huber_delta", hp.get("huber_delta", 1.0)))

                stage2_epochs = int(stage2_hp["epochs"])
                if stage2_epochs > 0:
                    logging.info(
                        f"Stage-2 hard-mining fine-tune: top_k={top_k}, mult={mult}, "
                        f"epochs={stage2_epochs}, lr={stage2_hp['learning_rate']}, loss={stage2_hp['loss_type']}"
                    )

                    stage2_criterion = build_loss(stage2_hp)
                    stage2_optimizer = build_optimizer(model, stage2_hp)
                    
                    stage2_steps_per_epoch = len(stage2_train_loader)
                    stage2_scheduler = build_noam_scheduler(stage2_optimizer, stage2_epochs, stage2_steps_per_epoch)

                    stage2_best_val_mae = float("inf")
                    stage2_best_epoch = -1

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
                            scheduler=stage2_scheduler,
                        )
                        tr_loss2 = float(train2_out['loss']) if isinstance(train2_out, dict) else float(train2_out)
                        v_loss2, vpred_s2, vtrue_s2, vids2 = evaluate_epoch(model, val_loader, stage2_criterion, device)

                        vpred2 = vpred_s2
                        vtrue2 = vtrue_s2
                        if label_median is not None and label_iqr is not None and vpred2.size and vtrue2.size:
                            vpred2 = inverse_scale_labels(vpred_s2, label_median, label_iqr)
                            vtrue2 = inverse_scale_labels(vtrue_s2, label_median, label_iqr)

                        if vpred2.size and vtrue2.size:
                            v_mae2, v_r2_2 = log_val_diagnostics(
                                e2, stage2_hp, vtrue2, vpred2, vids2, tracker=stage2_worst_tracker
                            )
                            logging.info(
                                f"Stage2 Epoch {e2+1}/{stage2_epochs}: "
                                f"TrainLoss={tr_loss2:.6f}, ValLoss={v_loss2:.6f}, ValR2={v_r2_2:.4f}, ValMAE={v_mae2:.6f}"
                            )
                        else:
                            v_mae2 = float("inf")
                            logging.info(
                                f"Stage2 Epoch {e2+1}/{stage2_epochs}: "
                                f"TrainLoss={tr_loss2:.6f}, ValLoss={v_loss2:.6f}, ValR2=nan, ValMAE=nan"
                            )

                        if v_mae2 < stage2_best_val_mae:
                            stage2_best_val_mae = float(v_mae2)
                            stage2_best_epoch = int(e2 + 1)
                            torch.save(model.state_dict(), STAGE2_MODEL_SAVE_PATH)

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

    final_ckpt_path = MODEL_SAVE_PATH
    if bool(hp.get("enable_hard_mining", False)) and os.path.exists(STAGE2_MODEL_SAVE_PATH):
        final_ckpt_path = STAGE2_MODEL_SAVE_PATH

    try:
        state = torch.load(final_ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(final_ckpt_path, map_location=device)
    model.load_state_dict(state)

    if bool(hp.get("save_feature_importance", True)):
        fi_split, fi_loader = _resolve_feature_importance_loader(
            data_list,
            train_indices,
            val_loader,
            test_loader,
            hp,
            skip_test=skip_test,
        )
        if fi_loader is None:
            logging.warning("Feature importance skipped: no evaluation split available.")
        else:
            fi_repeats = int(hp.get("feature_importance_repeats", 5) or 1)
            fi_seed = int(hp.get("feature_importance_seed", SEED) or SEED)
            fi_rows = compute_block_permutation_importance(
                model,
                fi_loader,
                device,
                label_median,
                label_iqr,
                repeats=fi_repeats,
                seed=fi_seed,
                split_name=fi_split,
            )
            if fi_rows:
                save_block_permutation_importance_csv(fi_rows, FEATURE_IMPORTANCE_CSV_PATH)
                result["feature_importance_csv"] = FEATURE_IMPORTANCE_CSV_PATH
                result["feature_importance_split"] = fi_split
                logging.info(
                    f"Saved block permutation feature importance to {FEATURE_IMPORTANCE_CSV_PATH} "
                    f"(split={fi_split}, repeats={fi_repeats})"
                )
            else:
                logging.warning("Feature importance skipped: prediction collection returned no samples.")

    if skip_test:
        return result

    if len(test_loader.dataset) == 0:
        logging.warning("Test dataset is empty.")
        return result

    id2smiles = _load_id_to_smiles_map() if bool(hp.get("include_smiles_in_worstk", True)) else {}
    final_k = int(hp.get("final_worst_k", hp.get("worst_k", 20)))

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


def main() -> None:
    parser = argparse.ArgumentParser()
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

    label_median, label_iqr = calculate_label_scaling_params(data_list, train_indices)
    if label_median is not None and label_iqr is not None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.savez(LABEL_SCALING_PARAMS_PATH, median=label_median, iqr=label_iqr)
        apply_label_scaling(data_list, label_median, label_iqr)

    g_mean, g_std = fit_global_standard_scaler(data_list, train_indices)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(GLOBAL_SCALING_PARAMS_PATH, mean=g_mean, std=g_std)
    clip_value = float(HYPERPARAMS.get("global_clip_value", 0.0))
    apply_global_scaling(data_list, g_mean, g_std, clip_value=clip_value if clip_value > 0 else None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

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

