import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from rdkit import Chem
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Batch
import traceback
import time
import os
import logging
from typing import Optional, List, Tuple

try:
    from data_processing import smiles_to_graph_data, initialize_feature_scaling, inverse_scale_labels
    from config import HYPERPARAMS
    from feature_configs import (
        TOTAL_FEATURE_DIMENSION, NUM_BOND_FEATURES, NUM_LINE_EDGE_FEATURES
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    exit(1)

class EdgeGatedConv_for_MP_BP(MessagePassing):
    def __init__(self, node_in_dim: int, edge_in_dim: int, out_dim: int):
        super().__init__(aggr='add')
        self.node_mlp = nn.Linear(node_in_dim + edge_in_dim, out_dim)
        self.edge_mlp = nn.Linear(node_in_dim + node_in_dim + edge_in_dim, out_dim)
        self.gate_mlp = nn.Linear(node_in_dim + edge_in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_mlp(torch.cat([x_i, edge_attr], dim=-1)))
        return gate * self.node_mlp(torch.cat([x_j, edge_attr], dim=-1))

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        new_x = aggr_out
        row, col = edge_index
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        new_edge_attr = self.edge_mlp(edge_input)
        return new_x, new_edge_attr

class LOHCGNN_for_MP_BP(nn.Module):
    def __init__(self, node_in_dim: int, edge_in_dim: int, line_edge_in_dim: int,
                 hidden_dim: int, num_layers: int, num_output_features: int, dropout_rate: float = 0.5):
        super().__init__()
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim)
        self.line_edge_embed = nn.Linear(line_edge_in_dim, hidden_dim)

        self.atom_conv_layers = nn.ModuleList([
            EdgeGatedConv_for_MP_BP(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.line_conv_layers = nn.ModuleList([
            EdgeGatedConv_for_MP_BP(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_output_features)
        )

    def forward(self, atom_data: Batch, line_data: Batch) -> torch.Tensor:
        h = self.node_embed(atom_data.x)
        e = self.edge_embed(atom_data.edge_attr)
        l = self.edge_embed(line_data.x)
        le = self.line_edge_embed(line_data.edge_attr)

        for atom_conv, line_conv in zip(self.atom_conv_layers, self.line_conv_layers):
            l_update, le_update = line_conv(l, line_data.edge_index, le)
            h_update, e_update = atom_conv(h, atom_data.edge_index, e)

            h = h + h_update
            e = e + e_update
            l = l + l_update
            le = le + le_update
            
        h_pooled = global_mean_pool(h, atom_data.batch)
        
        return self.mlp(h_pooled)

CONFIG_2ND = {
    'input_first_stage_results_file': 'promising_lohc_candidates.xlsx',
    'output_final_results_file': 'final_lohc_candidates.xlsx',
    'batch_size': HYPERPARAMS.get('batch_size', 32),
    'num_workers': 0,
    
    'mp_model_path': 'lohc_model_MP.pth',
    'mp_scaler_path': 'lohc_scaler_MP.npz',
    'bp_model_path': 'lohc_model_BP.pth',
    'bp_scaler_path': 'lohc_scaler_BP.npz',

    'max_melting_point': -20.0,
    'min_boiling_point': 200.0,
}

log_level = os.environ.get('LOGGING_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')

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

def predict_property(model: nn.Module, data_loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for data_pair in data_loader:
            (atom_de_b, line_de_b), (atom_hy_b, line_hy_b) = data_pair
            atom_batch = atom_hy_b.to(device)
            line_batch = line_hy_b.to(device)
            try:
                output = model(atom_batch, line_batch)
                all_predictions.append(output.cpu().numpy())
            except Exception as pred_err:
                logging.error(f"Error during prediction batch: {pred_err}")
                num_in_batch = atom_batch.num_graphs
                all_predictions.append(np.full((num_in_batch, 1), np.nan))
    
    return np.concatenate(all_predictions, axis=0) if all_predictions else np.array([])

def screen_second_stage():
    logging.info("Starting 2nd stage LOHC candidate screening (MP/BP for both molecules)...")
    start_time = time.time()

    try:
        df = pd.read_excel(CONFIG_2ND['input_first_stage_results_file'])
        logging.info(f"Loaded {len(df)} candidates from 1st stage screening.")
        if df.empty: return
    except FileNotFoundError:
        logging.error(f"Input file not found: {CONFIG_2ND['input_first_stage_results_file']}")
        return
    except Exception as e:
        logging.error(f"Error reading input file: {e}")
        return

    logging.info("Initializing feature scaling...")
    initialize_feature_scaling(TOTAL_FEATURE_DIMENSION)

    logging.info("Generating graph data for both Dehydrogenated and Hydrogenated molecules...")
    dehydro_graph_list, hydro_graph_list = [], []
    valid_indices = []
    for index, row in df.iterrows():
        smiles_de = row['Dehydrogenated_SMILES']
        smiles_hy = row['Hydrogenated_SMILES']
        
        dummy_labels = [0.0] 
        graph_pair_de = smiles_to_graph_data(smiles_de, dummy_labels)
        graph_pair_hy = smiles_to_graph_data(smiles_hy, dummy_labels)

        if graph_pair_de and graph_pair_hy:
            dehydro_graph_list.append((graph_pair_de, graph_pair_de))
            hydro_graph_list.append((graph_pair_hy, graph_pair_hy))
            valid_indices.append(index)
        else:
            logging.warning(f"Skipping candidate ID {row.get('ID', index)} due to graph generation failure.")
    
    if not valid_indices:
        logging.error("No valid graph data could be generated. Exiting.")
        return
        
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    dehydro_loader = DataLoader(dehydro_graph_list, batch_size=CONFIG_2ND['batch_size'], shuffle=False)
    hydro_loader = DataLoader(hydro_graph_list, batch_size=CONFIG_2ND['batch_size'], shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mp_min, mp_max = load_label_scaling_params(CONFIG_2ND['mp_scaler_path'])
    bp_min, bp_max = load_label_scaling_params(CONFIG_2ND['bp_scaler_path'])
    if mp_min is None or bp_min is None: return

    try:
        mp_model = LOHCGNN_for_MP_BP(
            node_in_dim=TOTAL_FEATURE_DIMENSION,
            edge_in_dim=NUM_BOND_FEATURES,
            line_edge_in_dim=NUM_LINE_EDGE_FEATURES,
            hidden_dim=HYPERPARAMS['hidden_dim'],
            num_layers=HYPERPARAMS['num_layers'],
            num_output_features=1,
            dropout_rate=HYPERPARAMS['dropout_rate']
        ).to(device)
        mp_model.load_state_dict(torch.load(CONFIG_2ND['mp_model_path'], map_location=device))
        
        bp_model = LOHCGNN_for_MP_BP(
            node_in_dim=TOTAL_FEATURE_DIMENSION,
            edge_in_dim=NUM_BOND_FEATURES,
            line_edge_in_dim=NUM_LINE_EDGE_FEATURES,
            hidden_dim=HYPERPARAMS['hidden_dim'],
            num_layers=HYPERPARAMS['num_layers'],
            num_output_features=1,
            dropout_rate=HYPERPARAMS['dropout_rate']
        ).to(device)
        bp_model.load_state_dict(torch.load(CONFIG_2ND['bp_model_path'], map_location=device))
        
    except Exception as e:
        logging.error(f"Failed to load MP/BP models: {e}")
        traceback.print_exc()
        return

    logging.info("Predicting MP/BP for Dehydrogenated molecules...")
    mp_de_scaled = predict_property(mp_model, dehydro_loader, device)
    bp_de_scaled = predict_property(bp_model, dehydro_loader, device)
    df_valid['Predicted_MP_Dehydro'] = inverse_scale_labels(mp_de_scaled, mp_min, mp_max).flatten()
    df_valid['Predicted_BP_Dehydro'] = inverse_scale_labels(bp_de_scaled, bp_min, bp_max).flatten()

    logging.info("Predicting MP/BP for Hydrogenated molecules...")
    mp_hy_scaled = predict_property(mp_model, hydro_loader, device)
    bp_hy_scaled = predict_property(bp_model, hydro_loader, device)
    df_valid['Predicted_MP_Hydro'] = inverse_scale_labels(mp_hy_scaled, mp_min, mp_max).flatten()
    df_valid['Predicted_BP_Hydro'] = inverse_scale_labels(bp_hy_scaled, bp_min, bp_max).flatten()
    
    logging.info("Filtering final candidates based on MP and BP of BOTH molecules...")
    final_df = df_valid.copy()
    
    prop_cols = ['Predicted_MP_Dehydro', 'Predicted_BP_Dehydro', 'Predicted_MP_Hydro', 'Predicted_BP_Hydro']
    final_df.dropna(subset=prop_cols, inplace=True)
    
    dehydro_liquid_mask = (final_df['Predicted_MP_Dehydro'] <= CONFIG_2ND['max_melting_point']) & \
                          (final_df['Predicted_BP_Dehydro'] >= CONFIG_2ND['min_boiling_point'])
                          
    hydro_liquid_mask = (final_df['Predicted_MP_Hydro'] <= CONFIG_2ND['max_melting_point']) & \
                        (final_df['Predicted_BP_Hydro'] >= CONFIG_2ND['min_boiling_point'])

    final_mask = dehydro_liquid_mask & hydro_liquid_mask
    
    logging.info(f"  Initial valid candidates: {len(df_valid)}")
    logging.info(f"  Candidates where Dehydro-LOHC is liquid: {dehydro_liquid_mask.sum()}")
    logging.info(f"  Candidates where Hydro-LOHC is liquid: {hydro_liquid_mask.sum()}")
    
    final_df = final_df[final_mask]
    logging.info(f"  Total {len(final_df)} candidates remaining after final MP/BP filters.")

    if not final_df.empty:
        logging.info(f"Saving final candidates to {CONFIG_2ND['output_final_results_file']}...")
        potential_col = [col for col in final_df.columns if 'Predicted_Standard oxidation potential' in col][0]
        capacity_col = [col for col in final_df.columns if 'Predicted_Capacity' in col][0]
        
        sorted_df = final_df.sort_values(
            by=[capacity_col, potential_col, 'Predicted_BP_Dehydro', 'Predicted_BP_Hydro'],
            ascending=[False, True, False, False] 
        )
        
        try:
            sorted_df.to_excel(CONFIG_2ND['output_final_results_file'], index=False, engine='openpyxl')
            logging.info("Final results saved successfully.")
        except Exception as e:
            logging.error(f"Error saving final results: {e}")
    else:
        logging.info("No candidates passed the 2nd stage filtering criteria.")

    end_time = time.time()
    logging.info(f"2nd stage screening finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    screen_second_stage()