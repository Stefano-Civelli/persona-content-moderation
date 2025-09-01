import argparse
import json
import warnings
from pathlib import Path
from multiprocessing import Pool
import itertools
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
)
from sklearn.decomposition import PCA, TruncatedSVD, NMF, FastICA
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from umap import UMAP
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def load_and_prepare_data(json_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load and prepare data from JSON file."""
    print(f"Loading data from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    results = data.get("results")
    if not results:
        raise ValueError("JSON file does not contain a 'results' key or the list is empty.")

    df = pd.DataFrame(results)
    pred_labels_df = df["predicted_labels"].apply(pd.Series)
    df = pd.concat([
        df.drop(["predicted_labels", "true_labels", "raw_prediction"], axis=1),
        pred_labels_df,
    ], axis=1)
    
    label_cols = list(pred_labels_df.columns)
    df = df.sort_values(by="item_id").reset_index(drop=True)
    
    # Handle missing values
    if df[label_cols].isnull().values.any():
        print("Warning: Missing values found in predicted labels. Filling with 'missing'.")
        df[label_cols] = df[label_cols].fillna("missing")
    
    # Convert boolean labels to string
    for col in label_cols:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(str)
    
    print(f"Loaded {df['item_id'].nunique()} unique items and {df['persona_id'].nunique()} unique personas.")
    return df, label_cols


def create_onehot_features(df: pd.DataFrame, label_cols: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """Create standard one-hot encoded features."""
    print("Creating one-hot encoded features...")
    
    persona_info = df[['persona_id', 'persona_pos']].drop_duplicates().sort_values('persona_id').reset_index(drop=True)
    num_personas = persona_info.shape[0]
    
    df_sorted = df.sort_values(by=['persona_id', 'item_id']).reset_index(drop=True)
    
    for col in label_cols:
        df_sorted[col] = df_sorted[col].astype('category')
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None)
    encoded_data = encoder.fit_transform(df_sorted[label_cols])
    feature_matrix = encoded_data.reshape(num_personas, -1)
    
    return feature_matrix, persona_info


def create_frequency_features(df: pd.DataFrame, label_cols: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """Create frequency-based features."""
    print("Creating frequency-based features...")
    
    persona_info = df[['persona_id', 'persona_pos']].drop_duplicates().sort_values('persona_id').reset_index(drop=True)
    
    # Create frequency mappings for each label
    freq_features = []
    for persona_id in sorted(df['persona_id'].unique()):
        persona_data = df[df['persona_id'] == persona_id]
        features = []
        
        for col in label_cols:
            # Global frequency of each value
            global_freq = df[col].value_counts(normalize=True).to_dict()
            # Persona-specific frequency
            persona_freq = persona_data[col].value_counts(normalize=True).to_dict()
            
            # For each unique value in this column, add its frequency
            for value in df[col].unique():
                features.append(global_freq.get(value, 0))
                features.append(persona_freq.get(value, 0))
        
        freq_features.append(features)
    
    return np.array(freq_features), persona_info


def create_aggregated_features(df: pd.DataFrame, label_cols: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """Create statistical aggregation features."""
    print("Creating aggregated features...")
    
    persona_info = df[['persona_id', 'persona_pos']].drop_duplicates().sort_values('persona_id').reset_index(drop=True)
    
    agg_features = []
    for persona_id in sorted(df['persona_id'].unique()):
        persona_data = df[df['persona_id'] == persona_id]
        features = []
        
        for col in label_cols:
            # For categorical data, use one-hot encoding and then aggregate
            col_data = persona_data[col]
            unique_values = df[col].unique()
            
            # Create one-hot for this column
            for value in unique_values:
                count = (col_data == value).sum()
                proportion = count / len(col_data)
                features.append(proportion)
            
            # Additional statistics
            features.append(col_data.nunique())  # Number of unique values
            features.append(len(col_data))  # Total count
        
        agg_features.append(features)
    
    return np.array(agg_features), persona_info


def create_interaction_features(df: pd.DataFrame, label_cols: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """Create interaction features between label columns."""
    print("Creating interaction features...")
    
    persona_info = df[['persona_id', 'persona_pos']].drop_duplicates().sort_values('persona_id').reset_index(drop=True)
    
    # Create interaction columns
    df_with_interactions = df.copy()
    for i, col1 in enumerate(label_cols):
        for j, col2 in enumerate(label_cols[i+1:], i+1):
            interaction_col = f'{col1}_x_{col2}'
            df_with_interactions[interaction_col] = (
                df_with_interactions[col1].astype(str) + '_' + 
                df_with_interactions[col2].astype(str)
            )
    
    # Get all columns (original + interaction)
    all_feature_cols = [col for col in df_with_interactions.columns if col not in ['persona_id', 'persona_pos', 'item_id']]
    
    # One-hot encode all features
    df_sorted = df_with_interactions.sort_values(by=['persona_id', 'item_id']).reset_index(drop=True)
    
    for col in all_feature_cols:
        df_sorted[col] = df_sorted[col].astype('category')
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None)
    encoded_data = encoder.fit_transform(df_sorted[all_feature_cols])
    
    num_personas = persona_info.shape[0]
    feature_matrix = encoded_data.reshape(num_personas, -1)
    
    return feature_matrix, persona_info


def apply_dimensionality_reduction(feature_matrix: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """Apply different dimensionality reduction methods."""
    print(f"Applying {method} dimensionality reduction...")
    
    if method == 'umap':
        reducer = UMAP(n_components=2, **kwargs)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, **kwargs)
    elif method == 'pca':
        reducer = PCA(n_components=2, **kwargs)
    elif method == 'ica':
        reducer = FastICA(n_components=2, **kwargs)
    elif method == 'mds':
        reducer = MDS(n_components=2, **kwargs)
    elif method == 'isomap':
        reducer = Isomap(n_components=2, **kwargs)
    elif method == 'lle':
        reducer = LocallyLinearEmbedding(n_components=2, **kwargs)
    elif method == 'lda':
        # For LDA, we need labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        labels = le.fit_transform(kwargs.get('labels', []))
        reducer = LinearDiscriminantAnalysis(n_components=2)
        return reducer.fit_transform(feature_matrix, labels)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return reducer.fit_transform(feature_matrix)


def evaluate_clustering(embedding: np.ndarray, persona_info: pd.DataFrame) -> dict:
    """Evaluate clustering quality using multiple metrics."""
    from sklearn.preprocessing import LabelEncoder
    
    # Use political position as ground truth
    le = LabelEncoder()
    true_labels = le.fit_transform(persona_info['persona_pos'])
    
    # Calculate metrics
    try:
        silhouette = silhouette_score(embedding, true_labels)
        calinski = calinski_harabasz_score(embedding, true_labels)
        davies_bouldin = davies_bouldin_score(embedding, true_labels)
        
        return {
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies_bouldin
        }
    except:
        return {
            'silhouette': -1,
            'calinski_harabasz': -1,
            'davies_bouldin': -1
        }


def run_single_experiment(args_tuple):
    """Run a single experiment with given parameters."""
    (feature_matrix, persona_info, feature_type, method, method_params, 
     run_id, base_output_path, model_name) = args_tuple
    
    print(f"Running: {feature_type} + {method} (run {run_id})")
    
    try:
        # Apply dimensionality reduction
        if method == 'lda':
            # For LDA, we need labels
            method_params['labels'] = persona_info['persona_pos']
        
        embedding = apply_dimensionality_reduction(feature_matrix, method, **method_params)
        
        # Evaluate clustering
        eval_metrics = evaluate_clustering(embedding, persona_info)
        
        # Create a copy of persona_info for this experiment
        persona_info_copy = persona_info.copy()
        persona_info_copy["dim1"] = embedding[:, 0]
        persona_info_copy["dim2"] = embedding[:, 1]
        
        # Generate plot
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Define colors
        corner_colors = {
            "top_left": "blue",
            "top_right": "green",
            "bottom_left": "purple",
            "bottom_right": "red",
        }
        
        sns.scatterplot(
            data=persona_info_copy,
            x="dim1",
            y="dim2",
            hue="persona_pos",
            palette=corner_colors,
            s=150,
            ax=ax,
            edgecolor="black",
            alpha=0.8,
        )
        
        # Add annotations
        for i, row in persona_info_copy.iterrows():
            ax.text(
                row["dim1"] + 0.05,
                row["dim2"] + 0.05,
                str(row["persona_id"]),
                fontsize=8,
                fontstyle="italic",
                color="gray",
            )
        
        # Title with metrics
        title = f"{feature_type} + {method} (run {run_id})\n"
        title += f"Silhouette: {eval_metrics['silhouette']:.3f} | "
        title += f"Calinski-Harabasz: {eval_metrics['calinski_harabasz']:.1f} | "
        title += f"Davies-Bouldin: {eval_metrics['davies_bouldin']:.3f}"
        
        ax.set_title(title, fontsize=14, weight="bold")
        ax.set_xlabel(f"{method.upper()} Dimension 1", fontsize=12)
        ax.set_ylabel(f"{method.upper()} Dimension 2", fontsize=12)
        ax.legend(title="Political Position", bbox_to_anchor=(1.05, 1), loc="upper left")
        
        plt.tight_layout()
        
        # Create output filename
        param_str = '_'.join([f"{k}{v}" for k, v in method_params.items() if k != 'labels'])
        if param_str:
            param_str = f"_{param_str}"
        
        output_filename = f"{feature_type}_{method}{param_str}_run{run_id}.png"
        output_path = base_output_path / output_filename
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        result = {
            'output_path': output_path,
            'feature_type': feature_type,
            'method': method,
            'method_params': method_params,
            'run_id': run_id,
            'metrics': eval_metrics
        }
        
        print(f"Completed: {feature_type} + {method} (run {run_id}) - Silhouette: {eval_metrics['silhouette']:.3f}")
        return result
        
    except Exception as e:
        print(f"Error in {feature_type} + {method} (run {run_id}): {str(e)}")
        return None


def main():
    """Main function to run comprehensive experiments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive visualization experiments with multiple feature engineering methods and dimensionality reduction techniques.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task_type", type=str, default="text", help="Type of task")
    parser.add_argument("--timestamp", type=str, default="20250703_150720")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--num_runs", type=int, default=2, help="Number of runs for each combination")
    parser.add_argument("--num_processes", type=int, default=90, help="Number of processes for multiprocessing")
    parser.add_argument("--save_results", action="store_true", help="Save results to JSON")
    
    args = parser.parse_args()
    
    MODEL_NAME = args.model.split("/")[-1]
    input_file = Path(f"data/results/{args.task_type}_classification/{MODEL_NAME}/{args.timestamp}.json")
    base_output_path = Path(f"images/comprehensive_experiments/{MODEL_NAME}")
    
    # Load and prepare data once
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    df, label_cols = load_and_prepare_data(input_file)
    print(f"Loaded {len(df)} rows of data.")
    print("="*70)
    print()
    
    # Define feature engineering methods
    feature_methods = {
        'onehot': create_onehot_features,
        'frequency': create_frequency_features,
        'aggregated': create_aggregated_features,
        'interaction': create_interaction_features,
    }
    
    # Define dimensionality reduction methods and their parameters
    reduction_methods = {
        'umap': [
            {'n_neighbors': 5, 'min_dist': 0.1, 'metric': 'euclidean'},
            {'n_neighbors': 5, 'min_dist': 0.1, 'metric': 'cosine'},
            {'n_neighbors': 5, 'min_dist': 0.1, 'metric': 'hamming'},
            {'n_neighbors': 10, 'min_dist': 0.3, 'metric': 'jaccard'},
            {'n_neighbors': 15, 'min_dist': 0.5, 'metric': 'dice'},
        ],
        'tsne': [
            {'perplexity': 30, 'learning_rate': 200},
            {'perplexity': 50, 'learning_rate': 200},
            {'perplexity': 10, 'learning_rate': 200},
        ],
        'pca': [{}],
        'ica': [{}],
        'mds': [{}],
        'isomap': [{'n_neighbors': 5}, {'n_neighbors': 10}],
        'lle': [{'n_neighbors': 5}, {'n_neighbors': 10}],
        'lda': [{}],  # Will use persona_pos as labels
    }
    
    # Create all feature matrices
    print("="*70)
    print("CREATING FEATURE MATRICES")
    print("="*70)
    
    feature_matrices = {}
    persona_info = None
    
    for feat_name, feat_func in feature_methods.items():
        print(f"Creating {feat_name} features...")
        feature_matrix, persona_info = feat_func(df, label_cols)
        feature_matrices[feat_name] = feature_matrix
        print(f"  {feat_name} shape: {feature_matrix.shape}")
    
    print("="*70)
    print()
    
    # Generate experiment combinations
    experiment_combinations = []
    for feat_name, feature_matrix in feature_matrices.items():
        for method_name, param_list in reduction_methods.items():
            for param_dict in param_list:
                for run_id in range(1, args.num_runs + 1):
                    experiment_combinations.append((
                        feature_matrix, persona_info, feat_name, method_name, 
                        param_dict, run_id, base_output_path, MODEL_NAME
                    ))
    
    print(f"Total experiments: {len(experiment_combinations)}")
    
    # Run experiments
    print("="*70)
    print("RUNNING EXPERIMENTS")
    print("="*70)
    
    if args.num_processes == 1:
        results = [run_single_experiment(args) for args in experiment_combinations]
    else:
        with Pool(processes=args.num_processes) as pool:
            results = pool.map(run_single_experiment, experiment_combinations)
    
    # Filter out failed experiments
    successful_results = [r for r in results if r is not None]
    
    print("="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    # Sort by best silhouette score
    successful_results.sort(key=lambda x: x['metrics']['silhouette'], reverse=True)
    
    print(f"Total successful experiments: {len(successful_results)}")
    print(f"Failed experiments: {len(experiment_combinations) - len(successful_results)}")
    print()
    
    print("Top 10 results by Silhouette score:")
    for i, result in enumerate(successful_results[:10]):
        print(f"{i+1:2d}. {result['feature_type']:12} + {result['method']:6} | "
              f"Silhouette: {result['metrics']['silhouette']:6.3f} | "
              f"Calinski-Harabasz: {result['metrics']['calinski_harabasz']:8.1f} | "
              f"Davies-Bouldin: {result['metrics']['davies_bouldin']:6.3f}")
    
    # Save results if requested
    if args.save_results:
        results_file = base_output_path / "experiment_results.json"
        with open(results_file, 'w') as f:
            # Convert results to JSON-serializable format
            json_results = []
            for result in successful_results:
                json_result = {
                    'feature_type': result['feature_type'],
                    'method': result['method'],
                    'method_params': result['method_params'],
                    'run_id': result['run_id'],
                    'metrics': result['metrics'],
                    'output_path': str(result['output_path'])
                }
                json_results.append(json_result)
            
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()