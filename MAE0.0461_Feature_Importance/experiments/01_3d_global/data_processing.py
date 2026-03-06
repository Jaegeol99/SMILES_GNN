from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import config as config_module
from config import CSV_INDEX_COL, CSV_PATH, CSV_SMILES_COL, CSV_TARGET_COL, MAX_SAMPLES, OUTPUT_DIR
from feature_configs import (
    FUNCTIONAL_GROUP_PATTERNS,
    GLOBAL_FEATURE_DIM,
    LAPLACIAN_PE_K,
    LINE_NODE_FEATURE_DIM,
    MAX_GASTEIGER_ABS,
    MAX_NEIGHBOR_COUNT,
    NEIGHBOR_ATOMS_IDX,
    NUM_BOND_FEATURES,
    NUM_3D_GLOBAL_FEATURES,
    NUM_LINE_EDGE_FEATURES,
    NUM_NEIGHBOR_FEATURES,
    TOTAL_FEATURE_DIMENSION,
    one_hot_atom_type,
    one_hot_degree,
    one_hot_formal_charge,
    one_hot_hybridization,
    one_hot_total_h,
    scale_count,
    scale_signed,
)


PREPROCESSED_DATA_CACHE_PATH: str = str(
    getattr(config_module, "PREPROCESSED_DATA_CACHE_PATH", os.path.join(OUTPUT_DIR, "qm9_preprocessed_cache.pt"))
)
USE_PREPROCESSED_CACHE: bool = bool(getattr(config_module, "USE_PREPROCESSED_CACHE", True))
REBUILD_PREPROCESSED_CACHE: bool = bool(getattr(config_module, "REBUILD_PREPROCESSED_CACHE", False))
PREPROCESS_LOG_EVERY: int = int(getattr(config_module, "PREPROCESS_LOG_EVERY", 5000))
PREPROCESSED_CACHE_VERSION: int = int(getattr(config_module, "PREPROCESSED_CACHE_VERSION", 1))


THREE_D_EMBED_SEED: int = 42


def _safe_float(value: float) -> float:
    try:
        val = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(val):
        return 0.0
    return val


def _embed_molecule_3d(mol: Chem.Mol) -> Optional[Chem.Mol]:
    mol_3d = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(THREE_D_EMBED_SEED)

    try:
        status = AllChem.EmbedMolecule(mol_3d, params)
    except Exception:
        return None

    if status != 0:
        return None

    try:
        AllChem.UFFOptimizeMolecule(mol_3d, maxIters=200)
    except Exception:
        pass
    return mol_3d


def _compute_3d_global_features(mol: Chem.Mol) -> np.ndarray:
    mol_3d = _embed_molecule_3d(mol)
    if mol_3d is None or mol_3d.GetNumConformers() == 0:
        return np.zeros((NUM_3D_GLOBAL_FEATURES,), dtype=np.float32)

    conf = mol_3d.GetConformer()
    heavy_atom_indices = [atom.GetIdx() for atom in mol_3d.GetAtoms() if atom.GetAtomicNum() > 1]
    coord_indices = heavy_atom_indices if len(heavy_atom_indices) >= 2 else list(range(mol_3d.GetNumAtoms()))

    coords = np.array(
        [
            [conf.GetAtomPosition(idx).x, conf.GetAtomPosition(idx).y, conf.GetAtomPosition(idx).z]
            for idx in coord_indices
        ],
        dtype=np.float32,
    )

    shape_features = [
        _safe_float(rdMolDescriptors.CalcRadiusOfGyration(mol_3d)),
        _safe_float(rdMolDescriptors.CalcAsphericity(mol_3d)),
        _safe_float(rdMolDescriptors.CalcEccentricity(mol_3d)),
        _safe_float(rdMolDescriptors.CalcInertialShapeFactor(mol_3d)),
        _safe_float(rdMolDescriptors.CalcNPR1(mol_3d)),
        _safe_float(rdMolDescriptors.CalcNPR2(mol_3d)),
        _safe_float(rdMolDescriptors.CalcSpherocityIndex(mol_3d)),
        _safe_float(rdMolDescriptors.CalcPMI1(mol_3d)),
        _safe_float(rdMolDescriptors.CalcPMI2(mol_3d)),
        _safe_float(rdMolDescriptors.CalcPMI3(mol_3d)),
    ]

    bond_lengths: List[float] = []
    for bond in mol_3d.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        if begin_atom.GetAtomicNum() == 1 or end_atom.GetAtomicNum() == 1:
            continue
        pos_u = conf.GetAtomPosition(bond.GetBeginAtomIdx())
        pos_v = conf.GetAtomPosition(bond.GetEndAtomIdx())
        bond_lengths.append(float((pos_u - pos_v).Length()))

    if not bond_lengths:
        for bond in mol_3d.GetBonds():
            pos_u = conf.GetAtomPosition(bond.GetBeginAtomIdx())
            pos_v = conf.GetAtomPosition(bond.GetEndAtomIdx())
            bond_lengths.append(float((pos_u - pos_v).Length()))

    if coords.shape[0] >= 2:
        delta = coords[:, None, :] - coords[None, :, :]
        dist_mat = np.linalg.norm(delta, axis=-1)
        upper_tri = dist_mat[np.triu_indices(coords.shape[0], k=1)]
    else:
        upper_tri = np.zeros((0,), dtype=np.float32)

    bond_arr = np.asarray(bond_lengths, dtype=np.float32)
    distance_features = [
        _safe_float(bond_arr.mean()) if bond_arr.size else 0.0,
        _safe_float(bond_arr.std()) if bond_arr.size else 0.0,
        _safe_float(upper_tri.mean()) if upper_tri.size else 0.0,
        _safe_float(upper_tri.std()) if upper_tri.size else 0.0,
        _safe_float(upper_tri.max()) if upper_tri.size else 0.0,
    ]

    return np.asarray(shape_features + distance_features, dtype=np.float32)


def compute_global_features(mol: Chem.Mol) -> torch.Tensor:
    try:
        desc_list = []
        for _, func in Descriptors.descList:
            try:
                val = float(func(mol))
            except Exception:
                val = 0.0
            desc_list.append(val)
        desc = np.array(desc_list, dtype=np.float32)
    except Exception:
        desc = np.zeros((len(Descriptors.descList),), dtype=np.float32)

    desc_3d = _compute_3d_global_features(mol)
    desc = np.concatenate([desc, desc_3d], axis=0).astype(np.float32, copy=False)
    desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return torch.from_numpy(desc).view(1, -1)


def _build_preprocessed_cache_metadata() -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "cache_version": int(PREPROCESSED_CACHE_VERSION),
        "csv_path": os.path.abspath(CSV_PATH),
        "max_samples": MAX_SAMPLES,
        "total_feature_dimension": int(TOTAL_FEATURE_DIMENSION),
        "line_node_feature_dim": int(LINE_NODE_FEATURE_DIM),
        "num_line_edge_features": int(NUM_LINE_EDGE_FEATURES),
        "global_feature_dim": int(GLOBAL_FEATURE_DIM),
        "laplacian_pe_k": int(LAPLACIAN_PE_K),
    }
    try:
        stat = os.stat(CSV_PATH)
        metadata["csv_size"] = int(stat.st_size)
        metadata["csv_mtime_ns"] = int(stat.st_mtime_ns)
    except OSError:
        metadata["csv_size"] = None
        metadata["csv_mtime_ns"] = None
    return metadata


def _load_preprocessed_cache() -> Optional[Tuple[List[Tuple[Data, Data]], int]]:
    if not USE_PREPROCESSED_CACHE:
        return None
    if REBUILD_PREPROCESSED_CACHE:
        logging.info("Skipping preprocessed cache because REBUILD_PREPROCESSED_CACHE=True.")
        return None
    if not os.path.exists(PREPROCESSED_DATA_CACHE_PATH):
        return None

    started_at = time.perf_counter()
    try:
        try:
            payload = torch.load(PREPROCESSED_DATA_CACHE_PATH, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(PREPROCESSED_DATA_CACHE_PATH, map_location="cpu")
    except Exception as exc:
        logging.warning(f"Failed to load preprocessed cache {PREPROCESSED_DATA_CACHE_PATH}: {exc}")
        return None

    if not isinstance(payload, dict):
        logging.warning("Ignoring preprocessed cache because its payload format is invalid.")
        return None

    expected_metadata = _build_preprocessed_cache_metadata()
    if payload.get("metadata") != expected_metadata:
        logging.info("Preprocessed cache is stale; rebuilding graph dataset from CSV.")
        return None

    data_list = payload.get("data_list")
    num_node_features = int(payload.get("num_node_features", TOTAL_FEATURE_DIMENSION))
    if not isinstance(data_list, list):
        logging.warning("Ignoring preprocessed cache because data_list is missing or invalid.")
        return None

    elapsed = time.perf_counter() - started_at
    logging.info(
        "Loaded preprocessed dataset cache from %s in %.2fs (%d molecules).",
        PREPROCESSED_DATA_CACHE_PATH,
        elapsed,
        len(data_list),
    )
    return data_list, num_node_features


def _save_preprocessed_cache(data_list: List[Tuple[Data, Data]], num_node_features: int) -> None:
    if not USE_PREPROCESSED_CACHE:
        return

    os.makedirs(os.path.dirname(PREPROCESSED_DATA_CACHE_PATH) or OUTPUT_DIR, exist_ok=True)
    payload = {
        "metadata": _build_preprocessed_cache_metadata(),
        "data_list": data_list,
        "num_node_features": int(num_node_features),
    }

    started_at = time.perf_counter()
    try:
        torch.save(payload, PREPROCESSED_DATA_CACHE_PATH)
    except Exception as exc:
        logging.warning(f"Failed to save preprocessed cache {PREPROCESSED_DATA_CACHE_PATH}: {exc}")
        return

    elapsed = time.perf_counter() - started_at
    logging.info("Saved preprocessed dataset cache to %s in %.2fs.", PREPROCESSED_DATA_CACHE_PATH, elapsed)


def compute_laplacian_pe(mol: Chem.Mol, k: int) -> np.ndarray:
    num_atoms = mol.GetNumAtoms()
    if num_atoms <= 1:
        return np.zeros((num_atoms, k), dtype=np.float32)

    adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    deg = np.sum(adj, axis=1)
    lap = np.diag(deg) - adj

    with np.errstate(divide="ignore"):
        d_inv_sqrt = 1.0 / np.sqrt(deg)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    lap_norm = np.diag(d_inv_sqrt) @ lap @ np.diag(d_inv_sqrt)

    _, evecs = np.linalg.eigh(lap_norm)
    pe = evecs[:, 1 : k + 1]
    if pe.shape[1] < k:
        pe = np.pad(pe, ((0, 0), (0, k - pe.shape[1])))
    return pe.astype(np.float32)


def get_pseudo_angle_cos(atom: Chem.Atom) -> float:
    hyb = atom.GetHybridization()
    if hyb == Chem.rdchem.HybridizationType.SP3:
        angle = 109.5
    elif hyb == Chem.rdchem.HybridizationType.SP2:
        angle = 120.0
    elif hyb == Chem.rdchem.HybridizationType.SP:
        angle = 180.0
    else:
        angle = 90.0
    return float(np.cos(np.radians(angle)))


def get_shared_ring_size(mol: Chem.Mol, bond_idx1: int, bond_idx2: int) -> List[float]:
    bond1 = mol.GetBondWithIdx(bond_idx1)
    bond2 = mol.GetBondWithIdx(bond_idx2)
    if not (bond1.IsInRing() and bond2.IsInRing()):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    ring_info = mol.GetRingInfo()
    shared_sizes = [len(ring) for ring in ring_info.BondRings() if bond_idx1 in ring and bond_idx2 in ring]
    if not shared_sizes:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    min_size = min(shared_sizes)
    if min_size == 3:
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if min_size == 4:
        return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    if min_size == 5:
        return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if min_size == 6:
        return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def smiles_to_graph_data(smiles: str, labels: List[float], mol_id: Optional[int] = None) -> Optional[Tuple[Data, Data]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    num_atoms = mol.GetNumAtoms()

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        for atom in mol.GetAtoms():
            atom.SetDoubleProp("_GasteigerCharge", 0.0)

    func_group_matches = {
        name: {idx for match in mol.GetSubstructMatches(pat) for idx in match}
        for name, pat in FUNCTIONAL_GROUP_PATTERNS.items()
    }

    atom_features: List[List[float]] = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        g_charge = float(atom.GetProp("_GasteigerCharge")) if atom.HasProp("_GasteigerCharge") else 0.0

        feat = []
        feat += one_hot_atom_type(atom.GetAtomicNum())
        feat += one_hot_hybridization(atom.GetHybridization())
        feat.append(float(atom.GetIsAromatic()))
        feat += one_hot_total_h(atom.GetTotalNumHs())
        feat += one_hot_formal_charge(atom.GetFormalCharge())
        feat.append(scale_signed(g_charge, MAX_GASTEIGER_ABS))
        feat += one_hot_degree(atom.GetDegree())
        feat.append(float(atom.IsInRing()))

        neighbors = [0.0] * NUM_NEIGHBOR_FEATURES
        for neighbor in atom.GetNeighbors():
            idx = NEIGHBOR_ATOMS_IDX.get(neighbor.GetAtomicNum(), None)
            if idx is not None:
                neighbors[idx] += 1.0
        feat += [scale_count(v, MAX_NEIGHBOR_COUNT) for v in neighbors]
        feat += [1.0 if i in func_group_matches[name] else 0.0 for name in FUNCTIONAL_GROUP_PATTERNS.keys()]
        atom_features.append(feat)

    x_atom = torch.tensor(atom_features, dtype=torch.float)
    lap_pe_tensor = torch.tensor(compute_laplacian_pe(mol, LAPLACIAN_PE_K), dtype=torch.float)

    directed_edges: List[Tuple[int, int, int]] = []
    edge_features: List[List[float]] = []
    atom_edge_index, atom_edge_attr = [], []

    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_idx = bond.GetIdx()
        bond_type = bond.GetBondType()
        bond_feat = [
            float(bond_type == Chem.rdchem.BondType.SINGLE),
            float(bond_type == Chem.rdchem.BondType.DOUBLE),
            float(bond_type == Chem.rdchem.BondType.TRIPLE),
            float(bond_type == Chem.rdchem.BondType.AROMATIC),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
        ]
        atom_edge_index.extend([[u, v], [v, u]])
        atom_edge_attr.extend([bond_feat, bond_feat])
        directed_edges.extend([(u, v, bond_idx), (v, u, bond_idx)])
        edge_features.extend([bond_feat, bond_feat])

    if atom_edge_index:
        edge_index = torch.tensor(atom_edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(atom_edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, NUM_BOND_FEATURES), dtype=torch.float)

    if edge_features:
        bond_feat_tensor = torch.tensor(edge_features, dtype=torch.float)
        line_src = torch.tensor([u for (u, _, _) in directed_edges], dtype=torch.long)
        line_dst = torch.tensor([v for (_, v, _) in directed_edges], dtype=torch.long)
        pe_u = lap_pe_tensor[line_src]
        pe_v = lap_pe_tensor[line_dst]
        x_line = torch.cat([x_atom[line_src], bond_feat_tensor, pe_u, pe_v], dim=1)
    else:
        x_line = torch.empty((0, LINE_NODE_FEATURE_DIM), dtype=torch.float)
        line_src = torch.empty((0,), dtype=torch.long)
        line_dst = torch.empty((0,), dtype=torch.long)

    if directed_edges:
        outgoing: List[List[int]] = [[] for _ in range(num_atoms)]
        for idx, (u, _, _) in enumerate(directed_edges):
            outgoing[u].append(idx)

        hyb_feat_by_atom = [one_hot_hybridization(mol.GetAtomWithIdx(a).GetHybridization()) for a in range(num_atoms)]
        line_edge_indices, line_edge_attrs = [], []

        for i, (u_i, v_i, bond_idx_i) in enumerate(directed_edges):
            hyb_feat = hyb_feat_by_atom[v_i]
            pseudo_cos = get_pseudo_angle_cos(mol.GetAtomWithIdx(v_i))

            for j in outgoing[v_i]:
                _, v_j, bond_idx_j = directed_edges[j]
                if v_j == u_i:
                    continue

                ring_feat = get_shared_ring_size(mol, bond_idx_i, bond_idx_j)
                conj_flow = [1.0] if (edge_features[i][4] == 1.0 and edge_features[j][4] == 1.0) else [0.0]
                le_feat = hyb_feat + edge_features[i] + edge_features[j] + [pseudo_cos] + ring_feat + conj_flow
                line_edge_indices.append([i, j])
                line_edge_attrs.append(le_feat)

        if line_edge_indices:
            line_edge_index = torch.tensor(line_edge_indices, dtype=torch.long).t().contiguous()
            line_edge_attr = torch.tensor(line_edge_attrs, dtype=torch.float)
        else:
            line_edge_index = torch.empty((2, 0), dtype=torch.long)
            line_edge_attr = torch.empty((0, NUM_LINE_EDGE_FEATURES), dtype=torch.float)
    else:
        line_edge_index = torch.empty((2, 0), dtype=torch.long)
        line_edge_attr = torch.empty((0, NUM_LINE_EDGE_FEATURES), dtype=torch.float)

    try:
        g = compute_global_features(mol)
    except Exception:
        g = torch.zeros((1, GLOBAL_FEATURE_DIM), dtype=torch.float)

    y = torch.tensor([labels], dtype=torch.float)
    mol_id_tensor = torch.tensor([int(mol_id) if mol_id is not None else -1], dtype=torch.long)

    atom_graph = Data(x=x_atom, edge_index=edge_index, edge_attr=edge_attr, y=y, g=g, mol_id=mol_id_tensor)
    line_graph = Data(
        x=x_line,
        edge_index=line_edge_index,
        edge_attr=line_edge_attr,
        y=y,
        src=line_src,
        dst=line_dst,
        g=g,
        mol_id=mol_id_tensor,
    )
    return atom_graph, line_graph


def load_and_preprocess_qm9_data() -> Tuple[List[Tuple[Data, Data]], int]:
    cached = _load_preprocessed_cache()
    if cached is not None:
        return cached

    started_at = time.perf_counter()
    df = pd.read_csv(CSV_PATH)
    if MAX_SAMPLES is not None:
        df = df.iloc[:MAX_SAMPLES].copy()

    total_rows = len(df)
    logging.info(
        "Building graph dataset from %s (%d rows). This is CPU-bound and can take several minutes on the first run.",
        CSV_PATH,
        total_rows,
    )

    data_list: List[Tuple[Data, Data]] = []
    for row_idx, row in enumerate(df.itertuples(index=False), start=1):
        smiles = getattr(row, CSV_SMILES_COL, None)
        target = getattr(row, CSV_TARGET_COL, None)
        mol_id = getattr(row, CSV_INDEX_COL, None)
        if mol_id is None:
            mol_id = int(row_idx)
        if not isinstance(smiles, str) or not smiles:
            continue
        if target is None or (isinstance(target, float) and np.isnan(target)):
            continue

        res = smiles_to_graph_data(smiles, [float(target)], mol_id=mol_id)
        if res is not None:
            data_list.append(res)

        if PREPROCESS_LOG_EVERY > 0 and (row_idx % PREPROCESS_LOG_EVERY == 0 or row_idx == total_rows):
            elapsed = time.perf_counter() - started_at
            logging.info(
                "Preprocessing molecules: %d/%d rows, %d valid graph pairs, elapsed %.2fs",
                row_idx,
                total_rows,
                len(data_list),
                elapsed,
            )

    elapsed = time.perf_counter() - started_at
    logging.info("Finished preprocessing %d rows into %d graph pairs in %.2fs.", total_rows, len(data_list), elapsed)
    _save_preprocessed_cache(data_list, TOTAL_FEATURE_DIMENSION)
    return data_list, TOTAL_FEATURE_DIMENSION


def calculate_global_scaling_params(
    data_list: List[Tuple[Data, Data]],
    train_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    xs = [
        data_list[idx][0].g.detach().cpu().numpy().reshape(-1, GLOBAL_FEATURE_DIM)
        for idx in train_indices
        if hasattr(data_list[idx][0], "g") and data_list[idx][0].g is not None
    ]
    if not xs:
        return np.zeros((GLOBAL_FEATURE_DIM,), dtype=np.float32), np.ones((GLOBAL_FEATURE_DIM,), dtype=np.float32)

    X = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
    std = np.where(X.std(axis=0) < 1e-12, 1.0, X.std(axis=0)).astype(np.float32, copy=False)
    return X.mean(axis=0).astype(np.float32, copy=False), std


def apply_global_scaling(
    data_list: List[Tuple[Data, Data]],
    mean: np.ndarray,
    std: np.ndarray,
    clip_value: Optional[float] = None,
) -> None:
    mean_t = torch.tensor(mean, dtype=torch.float).view(1, -1)
    std_t = torch.tensor(std, dtype=torch.float).view(1, -1)
    do_clip = clip_value is not None and float(clip_value) > 0
    clip_min, clip_max = (-float(clip_value), float(clip_value)) if do_clip else (None, None)

    for atom_g, line_g in data_list:
        if hasattr(atom_g, "g") and atom_g.g is not None:
            g = (atom_g.g - mean_t) / std_t
            atom_g.g = torch.clamp(g, clip_min, clip_max) if do_clip else g
        if hasattr(line_g, "g") and line_g.g is not None:
            g = (line_g.g - mean_t) / std_t
            line_g.g = torch.clamp(g, clip_min, clip_max) if do_clip else g


def calculate_label_scaling_params(
    data_list: List[Tuple[Data, Data]],
    train_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    train_labels = np.concatenate([data_list[i][0].y.numpy() for i in train_indices], axis=0)
    q1, q3 = np.nanpercentile(train_labels, [25, 75], axis=0)
    return np.nanmedian(train_labels, axis=0), np.where((q3 - q1) < 1e-9, 1.0, q3 - q1)


def apply_label_scaling(data_list: List[Tuple[Data, Data]], median: np.ndarray, iqr: np.ndarray) -> None:
    for atom_g, line_g in data_list:
        scaled_y = torch.tensor((atom_g.y.numpy() - median) / iqr, dtype=torch.float)
        atom_g.y, line_g.y = scaled_y, scaled_y


def inverse_scale_labels(scaled: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    return scaled * iqr + median


def create_dataloaders(
    all_data: List[Tuple[Data, Data]],
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader([all_data[i] for i in train_idx], batch_size=batch_size, shuffle=True),
        DataLoader([all_data[i] for i in val_idx], batch_size=batch_size, shuffle=False),
        DataLoader([all_data[i] for i in test_idx], batch_size=batch_size, shuffle=False),
    )
