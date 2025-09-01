import argparse
import json
import warnings
from pathlib import Path
from multiprocessing import Pool
import itertools
from functools import partial

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from umap import UMAP

# Suppress a common UMAP warning about sparse matrix formats
warnings.filterwarnings(
    "ignore", message=".*The 'nopython' keyword argument was not supplied.*"
)

# suppress OneHotEncoder warning
warnings.filterwarnings(
    "ignore", message=".*X does not have valid feature names, but OneHotEncoder was fitted with feature names.*"
)

warnings.filterwarnings(
    "ignore", message=".*gradient function is not yet implemented for jaccard distance metric; inverse_transform will be unavailable.*"
)


def load_and_prepare_data(json_path: Path) -> tuple[pd.DataFrame, list[str]]:
    print(f"Loading data from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    results = data.get("results")
    if not results:
        raise ValueError(
            "JSON file does not contain a 'results' key or the list is empty."
        )

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(results)
    #     item_id                                        true_labels                                   predicted_labels                                     raw_prediction  persona_id persona_pos
    # 0  133743  {'is_hate_speech': False, 'target_category': '...  {'is_hate_speech': False, 'target_category': '...  {\n  "is_hate_speech": "false",\n  "target_cat...      116780    leftmost
    # 1   28206  {'is_hate_speech': False, 'target_category': '...  {'is_hate_speech': False, 'target_category': '...  {\n  "is_hate_speech": "false",\n  "target_cat...      116780    leftmost
    # 2   47889  {'is_hate_speech': True, 'target_category': 'm...  {'is_hate_speech': True, 'target_category': 'm...  {\n  "is_hate_speech": "true",\n  "target_cate...      116780    leftmost

    # The 'predicted_labels' column is a dictionary. We expand it into separate columns.
    # e.g., predicted_labels: {'is_hate_speech': False, ...} -> is_hate_speech | ...
    #                                                             False        | ...
    pred_labels_df = pd.json_normalize(df["predicted_labels"])
    #     is_hate_speech target_category
    # 0           False            none
    # 1           False            none
    # 2            True  muslims/arabic

    df = pd.concat(
        [
            df.drop(["predicted_labels", "true_labels", "raw_prediction"], axis=1),
            pred_labels_df,
        ],
        axis=1,
    )
    
    # Identify the label columns we need to encode
    label_cols = list(pred_labels_df.columns)
    print(f"Identified prediction labels: {label_cols}")

    # Critical Step: Ensure a consistent order of items for all personas
    df = df.sort_values(by="item_id").reset_index(drop=True)

    # Check for missing data
    if df[label_cols].isnull().values.any():
        print(
            "Warning: Missing values found in predicted labels. Filling with 'missing'."
        )
        df[label_cols] = df[label_cols].fillna("missing")

    # Convert boolean labels to string to handle them uniformly with other categorical labels
    for col in label_cols:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(str)

    print(
        f"Loaded {df['item_id'].nunique()} unique items and {df['persona_id'].nunique()} unique personas."
    )
    return df, label_cols


def create_persona_feature_vectors(df: pd.DataFrame, label_cols: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Creates feature vectors for each persona using a highly efficient, vectorized approach.
    """
    print("Creating feature vectors for each persona (vectorized approach)...")
    
    # Get unique personas and their political compass positions
    persona_info = df[['persona_id', 'persona_pos']].drop_duplicates().sort_values('persona_id').reset_index(drop=True)
    num_personas = persona_info.shape[0]
    num_items = df['item_id'].nunique()

    # --- Vectorized One-Hot Encoding ---
    # Sort by persona then item to ensure a consistent order for reshaping later.
    df_sorted = df.sort_values(by=['persona_id', 'item_id']).reset_index(drop=True)

    for col in label_cols:
        df_sorted[col] = df_sorted[col].astype('category')
        
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None) # cuML needs sparse=False
    
    # This single call replaces all the loops from the original function.
    encoded_data = encoder.fit_transform(df_sorted[label_cols])
    
    # The output is a 2D array of shape (num_personas * num_items, total_encoded_features).
    # We reshape it into a 3D array (personas, items, features) and then flatten the last two dims.
    # This creates one long vector per persona.
    # The -1 automatically calculates the length of the feature vector per persona.
    feature_matrix = encoded_data.reshape(num_personas, -1)

    print(f"CPU feature matrix created with shape: {feature_matrix.shape}")
    return feature_matrix, persona_info


def run_single_experiment(args_tuple):
    """
    Run a single UMAP experiment with given parameters and save the embedding data.
    """
    feature_matrix, persona_info, n_neighbors, min_dist, run_id, base_output_path, model_name = args_tuple
    
    print(f"Running experiment: n_neighbors={n_neighbors}, min_dist={min_dist}, run={run_id}")
    
    # Apply UMAP
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="jaccard",  # Jaccard is often a good choice for binary/categorical data
        # No random_state set to allow for different results across runs
    )
    embedding = reducer.fit_transform(feature_matrix)
    
    # Create a copy of persona_info for this experiment
    persona_info_copy = persona_info.copy()
    persona_info_copy["umap1"] = embedding[:, 0]
    persona_info_copy["umap2"] = embedding[:, 1]
    
    # Add experiment metadata
    persona_info_copy["n_neighbors"] = n_neighbors
    persona_info_copy["min_dist"] = min_dist
    persona_info_copy["run_id"] = run_id
    persona_info_copy["model_name"] = model_name
    
    # Create output filename for CSV
    output_filename = f"embedding_data_n{n_neighbors}_dist{min_dist}_run{run_id}.csv"
    output_path = base_output_path / output_filename
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the embedding data
    persona_info_copy.to_csv(output_path, index=False)
    
    print(f"Saved embedding data to: {output_path}")
    return output_path


def main():
    """Main function to parse arguments and run multiple experiments."""
    parser = argparse.ArgumentParser(
        description="Generate persona embeddings from classification results using UMAP with multiple parameter combinations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task_type", type=str, default="text", help="Type of task"
    )  # text, img
    parser.add_argument(
        "--timestamp",
        type=str,
        default="20250713_103643",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-32B-Instruct",
        help="Model name/path",
    )
    parser.add_argument(
        "--n_neighbors_list",
        type=int,
        nargs='+',
        default=[5],
        help="List of n_neighbors values to test",
    )
    parser.add_argument(
        "--min_dist_list",
        type=float,
        nargs='+',
        default=[0.1, 0.5],
        help="List of min_dist values to test",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=2,
        help="Number of runs for each parameter combination",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=6,
        help="Number of processes to use for multiprocessing",
    )

    args = parser.parse_args()

    MODEL_NAME = args.model.split("/")[-1]

    input_file = Path(
        f"data/results/{args.task_type}_classification/{MODEL_NAME}/{args.timestamp}/final_results.json"
    )
    base_output_path = Path(
        f"data/results/behavioural_embeddings/{MODEL_NAME}/{args.timestamp}"
    )

    # --- Load and prepare data once ---
    print("="*70)
    print("LOADING DATA (This will be done only once)")
    print("="*70)
    df, label_cols = load_and_prepare_data(input_file)
    print(f"Loaded {len(df)} rows of data.")
    
    feature_matrix, persona_info = create_persona_feature_vectors(df, label_cols)
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Persona info shape: {persona_info.shape}")
    print("="*70)
    print()

    # --- Generate all parameter combinations ---
    param_combinations = list(itertools.product(
        args.n_neighbors_list,
        args.min_dist_list,
        range(1, args.num_runs + 1)  # Run IDs from 1 to num_runs
    ))
    
    print(f"Total experiments to run: {len(param_combinations)}")
    print("Parameter combinations:")
    for i, (n_neighbors, min_dist, run_id) in enumerate(param_combinations):
        print(f"  {i+1}. n_neighbors={n_neighbors}, min_dist={min_dist}, run={run_id}")
    print()

    # --- Prepare arguments for multiprocessing ---
    experiment_args = [
        (feature_matrix, persona_info, n_neighbors, min_dist, run_id, base_output_path, MODEL_NAME)
        for n_neighbors, min_dist, run_id in param_combinations
    ]

    # --- Run experiments using multiprocessing ---
    print("="*70)
    print("RUNNING EXPERIMENTS")
    print("="*70)
    
    with Pool(processes=args.num_processes) as pool:
        results = pool.map(run_single_experiment, experiment_args)
    
    print("="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)
    print("Generated embedding data files:")
    for result in results:
        print(f"  {result}")
    print(f"\nTotal data files generated: {len(results)}")


if __name__ == "__main__":
    main()