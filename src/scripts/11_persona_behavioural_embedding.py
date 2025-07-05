import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from umap import UMAP

# Suppress a common UMAP warning about sparse matrix formats
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword argument was not supplied.*")

def load_and_prepare_data(json_path: Path) -> tuple[pd.DataFrame, list[str]]:
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    results = data.get("results")
    if not results:
        raise ValueError("JSON file does not contain a 'results' key or the list is empty.")

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(results)

    # The 'predicted_labels' column is a dictionary. We expand it into separate columns.
    # e.g., predicted_labels: {'is_hate_speech': False, ...} -> is_hate_speech | ...
    #                                                             False        | ...
    pred_labels_df = df['predicted_labels'].apply(pd.Series)
    df = pd.concat([df.drop(['predicted_labels', 'true_labels', 'raw_prediction'], axis=1), pred_labels_df], axis=1)

    # Identify the label columns we need to encode
    label_cols = list(pred_labels_df.columns)
    print(f"Identified prediction labels: {label_cols}")

    # Critical Step: Ensure a consistent order of items for all personas
    df = df.sort_values(by='item_id').reset_index(drop=True)
    
    # Check for missing data
    if df[label_cols].isnull().values.any():
        print("Warning: Missing values found in predicted labels. Filling with 'missing'.")
        df[label_cols] = df[label_cols].fillna('missing')


    # Convert boolean labels to string to handle them uniformly with other categorical labels
    for col in label_cols:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(str)

    print(f"Loaded {df['item_id'].nunique()} unique items and {df['persona_id'].nunique()} unique personas.")
    return df, label_cols


def create_persona_feature_vectors(df: pd.DataFrame, label_cols: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    print("Creating feature vectors for each persona...")
    
    # Get unique personas and their political compass positions
    persona_info = df[['persona_id', 'persona_pos']].drop_duplicates().set_index('persona_id')
    persona_ids = persona_info.index.tolist()

    # Create one-hot encoders for each label type
    encoders = {}
    for col in label_cols:
        # Fit the encoder on all possible values in the dataset to ensure consistency
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(df[[col]])
        encoders[col] = encoder
        print(f"  - Label '{col}' has categories: {encoder.categories_[0]}")

    persona_vectors = []
    for persona_id in persona_ids:
        # Get all predictions for the current persona, already sorted by item_id
        persona_df = df[df['persona_id'] == persona_id]
        
        # This list will hold the one-hot encoded vectors for each prediction (for this persona)
        all_encoded_parts = []

        for _, row in persona_df.iterrows():
            # For each item, concatenate the one-hot vectors of its labels
            item_vector_parts = []
            for col in label_cols:
                # Reshape is needed as encoder expects a 2D array
                value_to_encode = np.array([[row[col]]])
                encoded_vector = encoders[col].transform(value_to_encode)
                item_vector_parts.append(encoded_vector.flatten())
            
            # Concatenate all label parts for this single item
            all_encoded_parts.extend(item_vector_parts)
            
        # Concatenate all item vectors into one giant vector for the persona
        full_persona_vector = np.concatenate(all_encoded_parts)
        persona_vectors.append(full_persona_vector)
    
    feature_matrix = np.array(persona_vectors)
    print(f"Successfully created feature matrix of shape: {feature_matrix.shape}")
    print(f"(Each of the {feature_matrix.shape[0]} personas is represented by a vector of length {feature_matrix.shape[1]})")

    return feature_matrix, persona_info.reset_index()


def reduce_and_visualize(
    feature_matrix: np.ndarray,
    persona_info: pd.DataFrame,
    output_path: Path,
    n_neighbors: int,
    min_dist: float,
    random_state: int
) -> None:
    print("Applying UMAP for dimensionality reduction...")
    reducer = UMAP(
        n_components=2, # 5
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric='jaccard' # Jaccard is often a good choice for binary/categorical data
    )
    embedding = reducer.fit_transform(feature_matrix)
    print(f"Reduced dimensions to {embedding.shape}")

    # Add embedding coordinates to the persona_info DataFrame for plotting
    persona_info['umap1'] = embedding[:, 0]
    persona_info['umap2'] = embedding[:, 1]
    
    # --- Visualization ---
    print(f"Generating plot and saving to {output_path}...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define a color palette for the corners
    corner_colors = {
        "top_left": "blue",
        "top_right": "green",
        "bottom_left": "purple",
        "bottom_right": "red"
    }

    sns.scatterplot(
        data=persona_info,
        x='umap1',
        y='umap2',
        hue='persona_pos',
        palette=corner_colors,
        s=150,  # size of points
        ax=ax,
        edgecolor='black',
        alpha=0.8
    )

    # Add annotations (persona_id) to each point
    for i, row in persona_info.iterrows():
        ax.text(
            row['umap1'] + 0.05, 
            row['umap2'] + 0.05, 
            row['persona_id'], 
            fontsize=9,
            fontstyle='italic',
            color='gray'
        )

    ax.set_title("Persona Clustering based on Harmful Content Predictions (UMAP)", fontsize=16, weight='bold')
    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    ax.legend(title="Political Compass Corner", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Done.")


def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(
        description="Visualize persona clusters from classification results using UMAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input_file",
        type=str,
        default="data/results/text_classification/Qwen2.5-32B-Instruct/20250703_150720.json",
        help="Path to the JSON file containing classification results."
    )
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        default="images/behavioural_embeddings",
        help="Path to save the output PNG plot."
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=3, #15
        help="UMAP: The size of local neighborhood (in terms of number of samples)."
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=0.1, #0.0
        help="UMAP: The effective minimum distance between embedded points."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for UMAP to ensure reproducibility."
    )

    args = parser.parse_args()

    

    # --- Execute the Action Plan ---
    df, label_cols = load_and_prepare_data(args.input_file)
    feature_matrix, persona_info = create_persona_feature_vectors(df, label_cols)
    reduce_and_visualize(
        feature_matrix,
        persona_info,
        Path(args.output_file),
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed
    )

if __name__ == "__main__":
    main()