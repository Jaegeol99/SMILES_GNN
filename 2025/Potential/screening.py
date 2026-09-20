import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from torch_geometric.loader import DataLoader
import traceback
import time
import os
import logging
from typing import Optional, List, Tuple

def hydrogenate_smiles_conservative(dehydro_smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(dehydro_smiles, sanitize=False)
    if mol is None:
        logging.warning(f"입력 SMILES가 유효하지 않습니다: {dehydro_smiles}")
        return None

    try:
        mol.UpdatePropertyCache(strict=False)
        rw_mol = Chem.RWMol(mol)

        for bond in rw_mol.GetBonds():
            if bond.GetBondType() != Chem.BondType.SINGLE:
                begin_atom_idx = bond.GetBeginAtomIdx()
                end_atom_idx = bond.GetEndAtomIdx()
                rw_mol.RemoveBond(begin_atom_idx, end_atom_idx)
                rw_mol.AddBond(begin_atom_idx, end_atom_idx, Chem.BondType.SINGLE)
        
        saturated_mol = rw_mol.GetMol()
        Chem.SanitizeMol(saturated_mol)
        
        return Chem.MolToSmiles(saturated_mol, canonical=True)

    except Exception as e:
        logging.error(f"수소화 또는 분자 검증 중 에러 발생 (SMILES: {dehydro_smiles}): {e}")
        return None

try:
    from data_processing import smiles_to_graph_data, initialize_feature_scaling, inverse_scale_labels
    from gnn_model import LOHCGNN
    from config import (
        HYPERPARAMS, PROPERTY_NAMES, LABEL_COLS, MODEL_SAVE_PATH, LABEL_SCALING_PARAMS_PATH
    )
    from feature_configs import (
        TOTAL_FEATURE_DIMENSION, NUM_BOND_FEATURES, NUM_LINE_EDGE_FEATURES
    )
except ImportError as e:
    print(f"Error importing required modules or config constants: {e}")
    print("Ensure data_processing.py, gnn_model.py, config.py, and feature_configs.py are accessible.")
    exit(1)

CONFIG = {
    'input_smiles_file': 'candidates.smi',
    'trained_model_path': MODEL_SAVE_PATH,
    'label_scaling_params_path': LABEL_SCALING_PARAMS_PATH,
    'output_results_file': 'promising_lohc_candidates.xlsx',
    'batch_size': HYPERPARAMS.get('batch_size', 32),
    'num_workers': 0,
    'max_potential': 0.12,
    'min_potential': 0.0,
    'min_capacity': 5.5,
}

log_level = os.environ.get('LOGGING_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO),
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_label_scaling_params(filepath):
    try:
        data = np.load(filepath)
        if 'min_vals' in data and 'max_vals' in data:
            logging.info(f"Loaded label scaling parameters from {filepath}")
            return data['min_vals'], data['max_vals']
        else:
            logging.error(f"Error: 'min_vals' or 'max_vals' not found in {filepath}")
            return None, None
    except FileNotFoundError:
        logging.error(f"Error: Label scaling parameter file not found at {filepath}")
        return None, None
    except Exception as e:
        logging.error(f"Error loading label scaling parameters: {e}")
        return None, None

def screen_candidates():
    logging.info("Starting LOHC candidate screening...")
    start_time = time.time()

    label_min_values, label_max_values = load_label_scaling_params(CONFIG['label_scaling_params_path'])
    if label_min_values is None or label_max_values is None:
        logging.error("Could not load label scaling parameters. Please run train.py first.")
        return

    logging.info(f"Label scaling parameters loaded:")
    logging.info(f"  Min values: {label_min_values}")
    logging.info(f"  Max values: {label_max_values}")
    label_range = label_max_values - label_min_values
    label_range[np.abs(label_range) < 1e-9] = 1.0
    logging.info(f"  Label ranges: {label_range}")

    logging.info(f"Loading SMILES from {CONFIG['input_smiles_file']}...")
    try:
        candidate_smiles_with_id = []
        with open(CONFIG['input_smiles_file'], 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    candidate_id, smiles_str = parts[0], parts[1]
                    if smiles_str:
                        candidate_smiles_with_id.append((candidate_id, smiles_str))
                elif len(parts) == 1 and parts[0]:
                     candidate_smiles_with_id.append((str(line_num + 1), parts[0]))
        logging.info(f"Loaded {len(candidate_smiles_with_id)} candidate SMILES with IDs.")
        if not candidate_smiles_with_id:
            logging.error("Error: No valid SMILES found in the input file.")
            return
    except FileNotFoundError:
        logging.error(f"Error: Input SMILES file not found at {CONFIG['input_smiles_file']}")
        return
    except Exception as e:
        logging.error(f"Error reading SMILES file: {e}")
        return

    logging.info("Initializing feature scaling...")
    try:
        initialize_feature_scaling(TOTAL_FEATURE_DIMENSION)
        logging.info(f"Feature scaling initialized with dimension: {TOTAL_FEATURE_DIMENSION}")
    except Exception as e:
         logging.error(f"An unexpected error occurred during feature scaling setup: {e}")
         return

    logging.info("Generating SMILES pairs and converting to graph data...")
    paired_graph_data_list = []
    valid_smiles_pairs_with_id = []
    processed_count = 0
    skipped_count = 0
    total_candidates = len(candidate_smiles_with_id)

    dummy_labels_original = [0.0] * len(LABEL_COLS)
    dummy_labels_scaled = ((np.array(dummy_labels_original) - label_min_values) / label_range).tolist()
    logging.info(f"Dummy labels scaling applied:")
    logging.info(f"  Original dummy labels: {dummy_labels_original}")  
    logging.info(f"  Scaled dummy labels: {dummy_labels_scaled}")

    for i, (candidate_id, smiles_de) in enumerate(candidate_smiles_with_id):
        if (i + 1) % 1000 == 0:
            logging.info(f"  Processed {i+1}/{total_candidates}...")
        if not isinstance(smiles_de, str) or not smiles_de:
            skipped_count += 1
            continue
        
        smiles_hy = hydrogenate_smiles_conservative(smiles_de)
        
        if not smiles_hy:
            logging.warning(f"Skipping candidate {candidate_id} ({smiles_de}) due to hydrogenation failure.")
            skipped_count += 1
            continue
            
        graph_pair_de = smiles_to_graph_data(smiles_de, dummy_labels_scaled)
        graph_pair_hy = smiles_to_graph_data(smiles_hy, dummy_labels_scaled)
        
        if graph_pair_de and graph_pair_hy:
            if processed_count == 0:
                 logging.info(f"First graph pair generated. Node feature dimension: {graph_pair_de[0].num_node_features}")
                 if graph_pair_de[0].num_node_features != TOTAL_FEATURE_DIMENSION:
                     logging.warning(f"Warning: Generated graph feature dimension ({graph_pair_de[0].num_node_features}) "
                                     f"differs from initialized dimension ({TOTAL_FEATURE_DIMENSION}). Check config/preprocessing logic.")
            paired_graph_data_list.append((graph_pair_de, graph_pair_hy))
            valid_smiles_pairs_with_id.append({'ID': candidate_id, 'Dehydrogenated_SMILES': smiles_de, 'Hydrogenated_SMILES': smiles_hy})
            processed_count += 1
        else:
            logging.warning(f"Skipping candidate {candidate_id} due to graph generation failure.")
            skipped_count += 1

    logging.info(f"Generated {processed_count} valid graph pairs. Skipped {skipped_count}.")
    if not paired_graph_data_list:
        logging.error("Error: No valid graph pairs generated. Check input SMILES quality and preprocessing logs.")
        return

    logging.info(f"Loading trained model from {CONFIG['trained_model_path']}...")
    try:
        model = LOHCGNN(
            node_in_dim=TOTAL_FEATURE_DIMENSION,
            edge_in_dim=NUM_BOND_FEATURES,
            line_edge_in_dim=NUM_LINE_EDGE_FEATURES,
            hidden_dim=HYPERPARAMS['hidden_dim'],
            num_layers=HYPERPARAMS['num_layers'],
            num_output_features=HYPERPARAMS['num_output_features'],
            dropout_rate=HYPERPARAMS['dropout_rate']
        )
        model.load_state_dict(torch.load(CONFIG['trained_model_path'], map_location=torch.device('cpu')))
        model.eval()
        logging.info("Model loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Error: Trained model file not found at {CONFIG['trained_model_path']}")
        return
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        traceback.print_exc()
        return

    logging.info("Predicting properties for graph pairs...")
    all_predictions_scaled = []
    data_loader = DataLoader(paired_graph_data_list, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device for prediction: {device}")
    model.to(device)

    batch_count = 0
    with torch.no_grad():
        for data_pair in data_loader:
            (atom_de_b, line_de_b), (atom_hy_b, line_hy_b) = data_pair
            atom_batch = atom_hy_b.to(device)
            atom_batch.x_de = atom_de_b.x.to(device)
            atom_batch.edge_index_de = atom_de_b.edge_index.to(device)
            atom_batch.edge_attr_de = atom_de_b.edge_attr.to(device)
            atom_batch.batch_de = atom_de_b.batch.to(device)
            line_batch = line_hy_b.to(device)
            line_batch.x_de = line_de_b.x.to(device)
            line_batch.edge_index_de = line_de_b.edge_index.to(device)
            line_batch.edge_attr_de = line_de_b.edge_attr.to(device)
            line_batch.batch_de = line_de_b.batch.to(device)
            try:
                output_scaled = model(atom_batch, line_batch)
                all_predictions_scaled.append(output_scaled.cpu().numpy())
                
                if batch_count == 0:
                    sample_scaled = output_scaled.cpu().numpy()[0]
                    sample_original = inverse_scale_labels(
                        sample_scaled.reshape(1, -1), 
                        label_min_values, 
                        label_max_values
                    )[0]
                    logging.info(f"First sample prediction debugging:")
                    logging.info(f"  Scaled output: {sample_scaled}")
                    logging.info(f"  Original scale output: {sample_original}")
                
                batch_count += 1
            except Exception as pred_err:
                 logging.error(f"Error during prediction batch: {pred_err}")
                 num_in_batch = atom_batch.num_graphs
                 all_predictions_scaled.append(np.full((num_in_batch, len(LABEL_COLS)), np.nan))
                 batch_count += 1

    if not all_predictions_scaled:
        logging.error("Error: No predictions were generated.")
        return

    predictions_scaled_np = np.concatenate(all_predictions_scaled, axis=0)
    logging.info("Inverse scaling predictions...")
    predictions_original_np = inverse_scale_labels(predictions_scaled_np, label_min_values, label_max_values)
    
    logging.info(f"Prediction statistics (original scale):")
    for i, prop_name in enumerate(PROPERTY_NAMES):
        prop_values = predictions_original_np[:, i]
        valid_mask = ~np.isnan(prop_values)
        if valid_mask.any():
            logging.info(f"  {prop_name}: min={np.min(prop_values[valid_mask]):.4f}, "
                        f"max={np.max(prop_values[valid_mask]):.4f}, "
                        f"mean={np.mean(prop_values[valid_mask]):.4f}")
        else:
            logging.info(f"  {prop_name}: All NaN values")

    results_df = pd.DataFrame(valid_smiles_pairs_with_id)
    if predictions_original_np.shape[1] != len(PROPERTY_NAMES):
         logging.error(f"Prediction dimension ({predictions_original_np.shape[1]}) mismatch with PROPERTY_NAMES count ({len(PROPERTY_NAMES)}). Cannot assign columns.")
         return
    for i, name in enumerate(PROPERTY_NAMES):
        results_df[f'Predicted_{name}'] = predictions_original_np[:, i]

    logging.info("Filtering candidates based on predicted properties...")
    
    filtered_df = results_df.copy()
    initial_count = len(filtered_df)
    logging.info(f"  Initial candidates: {initial_count}")

    pred_cols = [f'Predicted_{name}' for name in PROPERTY_NAMES]
    nan_mask = filtered_df[pred_cols].isna().any(axis=1)
    if nan_mask.any():
        filtered_df = filtered_df[~nan_mask]
        logging.info(f"  {initial_count - len(filtered_df)} candidates removed due to NaN predictions.")
        logging.info(f"  Candidates remaining: {len(filtered_df)}")

    potential_col = f'Predicted_{PROPERTY_NAMES[2]}'
    potential_mask = (filtered_df[potential_col] <= CONFIG['max_potential']) & \
                     (filtered_df[potential_col] >= CONFIG['min_potential'])
    
    pre_filter_count = len(filtered_df)
    filtered_df = filtered_df[potential_mask]
    logging.info(f"  {pre_filter_count - len(filtered_df)} candidates removed by Potential filter.")
    logging.info(f"  {len(filtered_df)} candidates remaining after Potential filter "
                 f"({CONFIG['min_potential']} <= Potential <= {CONFIG['max_potential']})")

    capacity_col = f'Predicted_{PROPERTY_NAMES[3]}'
    capacity_mask = filtered_df[capacity_col] >= CONFIG['min_capacity']
    
    pre_filter_count = len(filtered_df)
    filtered_df = filtered_df[capacity_mask]
    logging.info(f"  {pre_filter_count - len(filtered_df)} candidates removed by Capacity filter.")
    logging.info(f"  {len(filtered_df)} candidates remaining after Capacity filter "
                 f"(Predicted_Capacity >= {CONFIG['min_capacity']} wt%)")

    logging.info(f"Found {len(filtered_df)} promising candidates after all filters.")

    if not filtered_df.empty:
        logging.info(f"Saving promising candidates to {CONFIG['output_results_file']}...")
        try:
            sorted_df = filtered_df.sort_values(
                by=[capacity_col, potential_col], 
                ascending=[False, True]
            )
            sorted_df.to_excel(CONFIG['output_results_file'], index=False, engine='openpyxl')
            logging.info("Results saved successfully as Excel.")
        except ImportError:
            logging.warning("Warning: 'openpyxl' library not found. Install with 'pip install openpyxl' for Excel output.")
            csv_filename = CONFIG['output_results_file'].replace('.xlsx', '.csv')
            try:
                sorted_df.to_csv(csv_filename, index=False)
                logging.info(f"Results saved as CSV instead: {csv_filename}")
            except Exception as e_csv:
                logging.error(f"Error saving results as CSV: {e_csv}")
        except Exception as e_excel:
            logging.error(f"Error saving results to Excel: {e_excel}")
    else:
        logging.info("No candidates passed the filtering criteria.")

    end_time = time.time()
    logging.info(f"Screening finished in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    screen_candidates()