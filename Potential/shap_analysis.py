import os
import logging
from typing import List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import shap

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Draw

from feature_configs import (
    FUNCTIONAL_GROUP_SMARTS,
    NUM_BOND_FEATURES,
    NUM_LINE_EDGE_FEATURES,
)
from config import (
    DATA_FILE_PATH,
    DEHYDRO_SMILES_COL,
    HYDRO_SMILES_COL,
    LABEL_COLS,
    HYPERPARAMS,
    PROPERTY_NAMES,
    MODEL_SAVE_PATH,
    LABEL_SCALING_PARAMS_PATH,
    SHAP_CONFIG
)
from data_processing import inverse_scale_labels, load_and_preprocess_paired_data, apply_label_scaling
from gnn_model import LOHCGNN

shap_logger = logging.getLogger('shap')
shap_logger.setLevel(logging.CRITICAL)

PairedDataTuple = Tuple[Tuple[Data, Data], Tuple[Data, Data]]

# --- ▼▼▼ [수정됨] Feature 이름을 TPSA로 업데이트 ▼▼▼ ---
paper_feature_names = [
    'Aromatic', 'H Count', # 'Atom in Ring' 제거
    'B-Coordination', 'N-Coordination', 'O-Coordination', 'S-Coordination',
    'Methyl', 'Amino', 'Hydroxyl', 'Fluorine', 'Chlorine', 'Carboxyl',
    'Carbonyl', 'Amine', 'Imine', 'Nitrile',
    'MR', 'Rotatable Bonds', 'HBD', 'HBA',
    'Ring Count', 'Fsp³', 'TPSA', # TPSA 추가
]

bond_feature_names = [
    'Single Bond', 'Double Bond', 'Triple Bond', 'Aromatic Bond',
    'Conjugated Bond', 'Bond in Ring', 'Stereo Bond'
]

atom_features_unsat = [f'{name} (Unsat)' for name in paper_feature_names]
atom_features_sat = [f'{name} (Sat)' for name in paper_feature_names]
bond_features_unsat = [f'{name} (Unsat)' for name in bond_feature_names]
bond_features_sat = [f'{name} (Sat)' for name in bond_feature_names]

PAIRED_FEATURE_NAMES_FOR_SHAP = atom_features_unsat + atom_features_sat + bond_features_unsat + bond_features_sat
# --- ▲▲▲ [수정 완료] ▲▲▲ ---

class PairedGNNExplainerWrapper:
    def __init__(self, model: LOHCGNN,
                 template_data_pairs_scaled: List[PairedDataTuple],
                 label_min_max_values: Tuple[np.ndarray, np.ndarray],
                 num_atom_features: int,
                 num_edge_features: int):
        self.model = model
        self.model.eval()
        self.template_data_pairs_scaled = template_data_pairs_scaled
        self.label_min_values, self.label_max_values = label_min_max_values
        self.num_atom_features = num_atom_features
        self.num_edge_features = num_edge_features
        self.num_output_features = model.mlp[-1].out_features
        self.device = next(model.parameters()).device

        if not template_data_pairs_scaled:
            raise ValueError("Template data pairs list cannot be empty.")
        if num_atom_features <= 0 or num_edge_features <= 0:
            raise ValueError("Number of atom and edge features must be positive.")

    def _inverse_scale_labels(self, scaled_labels: np.ndarray) -> np.ndarray:
        return inverse_scale_labels(scaled_labels, self.label_min_values, self.label_max_values)

    def convert_paired_graphs_to_average_features(self, paired_data_list_scaled: List[PairedDataTuple]) -> np.ndarray:
        concat_features_list = []
        expected_total_features = 2 * (self.num_atom_features + self.num_edge_features)
        if not paired_data_list_scaled:
            return np.empty((0, expected_total_features))

        for (data_de_pair, data_hy_pair) in paired_data_list_scaled:
            data_de, line_de = data_de_pair
            data_hy, line_hy = data_hy_pair
            
            avg_atom_feat_de = data_de.x.mean(dim=0).cpu().numpy() if data_de.x is not None and data_de.x.numel() > 0 else np.zeros(self.num_atom_features)
            avg_atom_feat_hy = data_hy.x.mean(dim=0).cpu().numpy() if data_hy.x is not None and data_hy.x.numel() > 0 else np.zeros(self.num_atom_features)
            
            avg_edge_feat_de = line_de.x.mean(dim=0).cpu().numpy() if line_de.x is not None and line_de.x.numel() > 0 else np.zeros(self.num_edge_features)
            avg_edge_feat_hy = line_hy.x.mean(dim=0).cpu().numpy() if line_hy.x is not None and line_hy.x.numel() > 0 else np.zeros(self.num_edge_features)

            concat_features_list.append(np.concatenate([avg_atom_feat_de, avg_atom_feat_hy, avg_edge_feat_de, avg_edge_feat_hy]))

        return np.array(concat_features_list) if concat_features_list else np.empty((0, expected_total_features))

    def predict_from_average_features(self, X_concat_avg_features: np.ndarray) -> np.ndarray:
        num_samples = X_concat_avg_features.shape[0]
        expected_feature_dim = 2 * (self.num_atom_features + self.num_edge_features)

        if X_concat_avg_features.ndim == 1:
            X_concat_avg_features = X_concat_avg_features.reshape(1, -1)
        if X_concat_avg_features.shape[1] != expected_feature_dim:
            return np.full((num_samples, self.num_output_features), np.nan)

        num_templates = len(self.template_data_pairs_scaled)
        if num_templates == 0:
            return np.full((num_samples, self.num_output_features), np.nan)

        batch_de_list, batch_hy_list = [], []
        for i in range(num_samples):
            concat_feat = X_concat_avg_features[i]
            
            avg_atom_feat_de_np = concat_feat[:self.num_atom_features]
            avg_atom_feat_hy_np = concat_feat[self.num_atom_features : 2 * self.num_atom_features]
            avg_edge_feat_de_np = concat_feat[2 * self.num_atom_features : 2 * self.num_atom_features + self.num_edge_features]
            avg_edge_feat_hy_np = concat_feat[2 * self.num_atom_features + self.num_edge_features:]

            template_idx = i % num_templates
            (template_de_pair, template_hy_pair) = self.template_data_pairs_scaled[template_idx]
            template_de, template_line_de = template_de_pair
            template_hy, template_line_hy = template_hy_pair

            num_nodes_de = template_de.num_nodes if template_de.num_nodes > 0 else 1
            num_nodes_hy = template_hy.num_nodes if template_hy.num_nodes > 0 else 1
            num_edges_de = template_line_de.num_nodes if template_line_de.num_nodes > 0 else 1
            num_edges_hy = template_line_hy.num_nodes if template_line_hy.num_nodes > 0 else 1
            
            new_data_de = Data(x=torch.tensor(avg_atom_feat_de_np, dtype=torch.float).repeat(num_nodes_de, 1),
                               edge_index=template_de.edge_index, edge_attr=template_de.edge_attr, num_nodes=num_nodes_de)
            new_data_hy = Data(x=torch.tensor(avg_atom_feat_hy_np, dtype=torch.float).repeat(num_nodes_hy, 1),
                               edge_index=template_hy.edge_index, edge_attr=template_hy.edge_attr, num_nodes=num_nodes_hy)
            
            new_line_de = Data(x=torch.tensor(avg_edge_feat_de_np, dtype=torch.float).repeat(num_edges_de, 1),
                               edge_index=template_line_de.edge_index, edge_attr=template_line_de.edge_attr)
            new_line_hy = Data(x=torch.tensor(avg_edge_feat_hy_np, dtype=torch.float).repeat(num_edges_hy, 1),
                               edge_index=template_line_hy.edge_index, edge_attr=template_line_hy.edge_attr)
            
            batch_de_list.append((new_data_de, new_line_de))
            batch_hy_list.append((new_data_hy, new_line_hy))

        if not batch_de_list:
            return np.full((num_samples, self.num_output_features), np.nan)
        
        loader_input = [((de_atom, de_line), (hy_atom, hy_line)) for (de_atom, de_line), (hy_atom, hy_line) in zip(batch_de_list, batch_hy_list)]
        temp_loader = DataLoader(loader_input, batch_size=len(loader_input))

        all_predictions_scaled_list = []
        with torch.no_grad():
            for data_pair in temp_loader:
                (atom_de_b, line_de_b), (atom_hy_b, line_hy_b) = data_pair
                
                atom_batch = atom_hy_b.to(self.device)
                atom_batch.x_de = atom_de_b.x.to(self.device)
                atom_batch.edge_index_de = atom_de_b.edge_index.to(self.device)
                atom_batch.edge_attr_de = atom_de_b.edge_attr.to(self.device)
                atom_batch.batch_de = atom_de_b.batch.to(self.device)

                line_batch = line_hy_b.to(self.device)
                line_batch.x_de = line_de_b.x.to(self.device)
                line_batch.edge_index_de = line_de_b.edge_index.to(self.device)
                line_batch.edge_attr_de = line_de_b.edge_attr.to(self.device)
                line_batch.batch_de = line_de_b.batch.to(self.device)

                predictions_scaled_batch = self.model(atom_batch, line_batch)
                all_predictions_scaled_list.append(predictions_scaled_batch.cpu().numpy())

        predictions_scaled = np.concatenate(all_predictions_scaled_list, axis=0)
        predictions_original_scale = self._inverse_scale_labels(predictions_scaled)
        return predictions_original_scale

def generate_shap_barplot(shap_values: np.ndarray, feature_values: np.ndarray, feature_names_list: List[str], property_name: str, save_dir: str):
    num_features_to_display = SHAP_CONFIG.get('max_display_features', 20)
    plt.figure(figsize=(12, num_features_to_display * 0.6 + 2))
    
    shap.summary_plot(shap_values, feature_values, feature_names=feature_names_list, show=False, plot_type='bar', max_display=num_features_to_display)
    
    plt.xlabel("mean(|SHAP value|)", fontsize=18, fontweight='bold')
    
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.subplots_adjust(left=0.42)
    
    safe_property_name = property_name.replace(" ", "-").replace("/", "-")
    file_path = os.path.join(save_dir, f"SHAP-Bar-{safe_property_name}.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    logging.info(f"  - Saved SHAP bar plot to {file_path}")

def generate_shap_beeswarm_plot(shap_values: np.ndarray, feature_values: np.ndarray, feature_names_list: List[str], property_name: str, save_dir: str, xlim: Tuple[float, float] = None):
    num_features_to_display = SHAP_CONFIG.get('max_display_features', 20)
    plt.figure(figsize=(12, num_features_to_display * 0.6 + 2))
    
    shap.summary_plot(shap_values, feature_values, feature_names=feature_names_list, show=False, plot_type='dot', max_display=num_features_to_display)
    
    plt.xlabel("SHAP value", fontsize=18, fontweight='bold')

    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=18)
    
    if xlim:
        plt.xlim(xlim)
    
    fig = plt.gcf()
    if len(fig.axes) > 1 and fig.axes[-1] != ax:
        cbar_ax = fig.axes[-1]
        cbar_ax.set_ylabel('Feature value', fontsize=18, fontweight='bold')
        cbar_ax.tick_params(labelsize=16)
        
    plt.subplots_adjust(left=0.42)
    
    safe_property_name = property_name.replace(" ", "-").replace("/", "-")
    file_path = os.path.join(save_dir, f"SHAP-Beeswarm-{safe_property_name}.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    logging.info(f"  - Saved SHAP beeswarm plot to {file_path}")

def run_shap_analysis(model: LOHCGNN,
                      test_data_tuples_scaled: List[PairedDataTuple],
                      test_smiles_pairs: List[Tuple[str, str]],
                      all_feature_names: List[str],
                      target_property_names: List[str],
                      label_min_max_vals: Tuple[np.ndarray, np.ndarray],
                      num_node_features_single_graph: int,
                      num_edge_features_single_graph: int,
                      output_plot_dir: str = "."):
    os.makedirs(output_plot_dir, exist_ok=True)

    num_samples_for_shap = min(SHAP_CONFIG['max_samples'], len(test_data_tuples_scaled))
    
    if len(test_data_tuples_scaled) > num_samples_for_shap:
        sample_indices = np.random.choice(len(test_data_tuples_scaled), num_samples_for_shap, replace=False)
        samples_to_explain_scaled = [test_data_tuples_scaled[i] for i in sample_indices]
    else:
        samples_to_explain_scaled = test_data_tuples_scaled

    if not samples_to_explain_scaled:
        logging.error("No samples available for SHAP analysis.")
        return

    explainer_wrapper = PairedGNNExplainerWrapper(
        model, 
        samples_to_explain_scaled, 
        label_min_max_vals, 
        num_node_features_single_graph,
        num_edge_features_single_graph
    )
    X_to_explain_avg_features = explainer_wrapper.convert_paired_graphs_to_average_features(samples_to_explain_scaled)
    
    if X_to_explain_avg_features.size == 0:
        logging.error("Failed to convert graph data to feature vectors for SHAP analysis.")
        return
    if len(all_feature_names) != X_to_explain_avg_features.shape[1]:
        logging.error(f"Feature name count ({len(all_feature_names)}) does not match feature vector dimension ({X_to_explain_avg_features.shape[1]}).")
        return

    def shap_prediction_function(X_subset_avg_features: np.ndarray) -> np.ndarray:
        return explainer_wrapper.predict_from_average_features(X_subset_avg_features)

    num_background_samples = min(100, X_to_explain_avg_features.shape[0])
    background_data_avg_features = shap.sample(X_to_explain_avg_features, num_background_samples)

    logging.info(f"Starting SHAP analysis for {len(target_property_names)} properties...")
    
    all_shap_values = {}
    for prop_name in target_property_names:
        try:
            prop_idx = PROPERTY_NAMES.index(prop_name)
        except ValueError:
            logging.warning(f"Property '{prop_name}' not found in the model's output list defined in config.py. Skipping.")
            continue

        logging.info(f"  - Analyzing property: {prop_name} (at index {prop_idx})")
        
        def predict_single_target_property(X_subset_avg_features: np.ndarray) -> np.ndarray:
            all_predictions = shap_prediction_function(X_subset_avg_features)
            return all_predictions if all_predictions.ndim == 1 else all_predictions[:, prop_idx]

        kernel_explainer = shap.KernelExplainer(predict_single_target_property, background_data_avg_features)
        shap_values_for_property = kernel_explainer.shap_values(X_to_explain_avg_features, nsamples=SHAP_CONFIG['nsamples'])
        all_shap_values[prop_name] = shap_values_for_property

    if not all_shap_values:
        logging.error("SHAP analysis failed to produce any values.")
        return

    energy_prop_names = ['Dehydrogenated energy', 'Hydrogenated energy']
    energy_shap_values = [vals for name, vals in all_shap_values.items() if name in energy_prop_names]
    
    xlim_map = {
        'Standard oxidation potential': (-0.2, 0.2),
        'Capacity': (-1.0, 1.0)
    }
    
    if energy_shap_values:
        global_min_shap = min(np.min(vals) for vals in energy_shap_values)
        global_max_shap = max(np.max(vals) for vals in energy_shap_values)
        max_abs_shap = max(abs(global_min_shap), abs(global_max_shap))
        padding = max_abs_shap * 0.1
        energy_xlim = (-max_abs_shap - padding, max_abs_shap + padding)
        logging.info(f"Energy properties SHAP range calculated. Using shared x-axis limits: {energy_xlim}")
        for name in energy_prop_names:
            if name in all_shap_values:
                xlim_map[name] = energy_xlim

    for prop_name, shap_values_for_property in all_shap_values.items():
        logging.info(f"  - Generating plots and reports for {prop_name}")
        
        current_xlim = xlim_map.get(prop_name)
        
        generate_shap_barplot(shap_values_for_property, X_to_explain_avg_features, all_feature_names, prop_name, output_plot_dir)
        
        generate_shap_beeswarm_plot(
            shap_values_for_property, 
            X_to_explain_avg_features, 
            all_feature_names, 
            prop_name, 
            output_plot_dir, 
            xlim=current_xlim
        )

    logging.info("SHAP analysis completed.")

def main_shap_analysis_pipeline():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        label_scaling_params = np.load(LABEL_SCALING_PARAMS_PATH)
        label_min_max_values = (label_scaling_params['min_vals'], label_scaling_params['max_vals'])
        logging.info(f"Successfully loaded label scaling parameters from {LABEL_SCALING_PARAMS_PATH}")
    except Exception as e:
        logging.error(f"Could not load label scaling parameters from '{LABEL_SCALING_PARAMS_PATH}'. Error: {e}")
        return

    import pandas as pd
    try:
        df = pd.read_excel(DATA_FILE_PATH)
    except Exception as e:
        logging.error(f"Could not load original SMILES from {DATA_FILE_PATH}. Error: {e}")
        return

    paired_data_list, num_node_features = load_and_preprocess_paired_data(DATA_FILE_PATH, DEHYDRO_SMILES_COL, HYDRO_SMILES_COL, LABEL_COLS)
    
    if not paired_data_list:
        logging.error("Data loading resulted in an empty list. Cannot proceed.")
        return
        
    if num_node_features <= 0:
        logging.error("Feature extraction resulted in zero features. Cannot proceed.")
        return
        
    if len(paper_feature_names) != num_node_features:
        logging.warning(f"Mismatch between number of generated features ({num_node_features}) and defined feature names ({len(paper_feature_names)}).")

    apply_label_scaling(paired_data_list, label_min_max_values[0], label_min_max_values[1])
    
    all_indices = list(range(len(paired_data_list)))
    try:
        train_indices, temp_indices = train_test_split(
            all_indices,
            test_size=HYPERPARAMS['test_split_ratio'],
            random_state=HYPERPARAMS['random_state']
        )
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=0.5,
            random_state=HYPERPARAMS['random_state']
        )
    except ValueError as e:
        logging.error(f"Train-test split failed. Not enough data. Error: {e}")
        return

    test_data_scaled_tuples = [paired_data_list[i] for i in test_indices]
    
    valid_df = df.dropna(subset=[DEHYDRO_SMILES_COL, HYDRO_SMILES_COL] + LABEL_COLS).reset_index(drop=True)
    original_smiles_pairs = list(zip(valid_df[DEHYDRO_SMILES_COL], valid_df[HYDRO_SMILES_COL]))
    test_smiles_pairs_list = [original_smiles_pairs[i] for i in test_indices if i < len(original_smiles_pairs)]

    if not test_data_scaled_tuples:
        logging.error("Test dataset is empty after splitting. Cannot perform SHAP analysis.")
        return
    logging.info(f"Test set created with {len(test_data_scaled_tuples)} samples for SHAP analysis.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LOHCGNN(node_in_dim=num_node_features, edge_in_dim=NUM_BOND_FEATURES, line_edge_in_dim=NUM_LINE_EDGE_FEATURES,
                    hidden_dim=HYPERPARAMS['hidden_dim'], num_layers=HYPERPARAMS['num_layers'],
                    num_output_features=len(LABEL_COLS), dropout_rate=HYPERPARAMS['dropout_rate'])
    
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        logging.info(f"Successfully loaded model weights from {MODEL_SAVE_PATH}")
    except Exception as e:
        logging.error(f"Failed to load model from '{MODEL_SAVE_PATH}'. Error: {e}")
        return

    model.to(device)
    model.eval()

    # --- ▼▼▼ [수정됨] 분석 대상을 'Standard oxidation potential'과 'Capacity'로 한정 ▼▼▼ ---
    properties_to_analyze = ['Standard oxidation potential', 'Capacity']
    # --- ▲▲▲ [수정 완료] ▲▲▲ ---
    
    run_shap_analysis(model, test_data_scaled_tuples, test_smiles_pairs_list, PAIRED_FEATURE_NAMES_FOR_SHAP,
                      properties_to_analyze, label_min_max_values, num_node_features,
                      NUM_BOND_FEATURES,
                      output_plot_dir="shap_analysis_results")

if __name__ == "__main__":
    plt.rcParams['axes.unicode_minus'] = False
    main_shap_analysis_pipeline()