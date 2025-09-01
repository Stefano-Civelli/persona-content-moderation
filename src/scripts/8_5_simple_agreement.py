import argparse
import json
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import os
from utils.util import (
    load_config,
    get_model_config,
)
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Manager

DATASET_NAME = None

positions_text_labels = {
    "top_left": "Top-L",
    "top_right": "Top-R",
    "bottom_left": "Bot-L",
    "bottom_right": "Bot-R",
    "leftmost": "Left",
    "rightmost": "Right",
}


def save_pairwise_comparisons(pairwise_results, output_path):
    df = pd.DataFrame(pairwise_results, columns=["Persona_1", "Persona_2", "Agreement"])
    df.to_csv(output_path, index=False)
    print(f"Pairwise comparison results saved to {output_path}")


def precompute_data_structures(df, positions):
    """Pre-compute all necessary data structures for faster agreement computation"""
    labels = list(df.columns[4:])
    
    # Group data by position and persona for O(1) lookup
    position_data = {}
    persona_predictions = {}
    
    for pos in positions:
        pos_df = df[df["persona_pos"] == pos]
        position_data[pos] = {
            'df': pos_df,
            'personas': pos_df["persona_id"].unique() if not pos_df.empty else []
        }
        
        # Pre-compute predictions for each persona-label combination
        for persona in position_data[pos]['personas']:
            key = (pos, persona)
            persona_predictions[key] = {}
            persona_df = pos_df[pos_df["persona_id"] == persona]
            
            for label in labels:
                persona_predictions[key][label] = persona_df[label].values
    
    return labels, position_data, persona_predictions

def plot_agreement_matrix(
    agreement_df,
    positions,
    output_path,
    figsize=(8, 6),
    cmap="YlOrRd",
    vmin=None,
    vmax=None,
    tick_size=25,
    annot_size=22,
    title=None,
):

    agreement_matrix = agreement_df.to_numpy()

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(agreement_matrix), k=1)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # If vmin/vmax not provided, set them based on data
    if vmin is None:
        vmin = np.floor(np.min(agreement_matrix) * 100) / 100
    if vmax is None:
        vmax = np.ceil(np.max(agreement_matrix) * 100) / 100

    # Calculate tick locations for increments of 0.02
    # Use the exact vmin and vmax values for the first and last ticks
    ticks = np.arange(vmin, vmax + 0.02, 0.02)
    # Ensure the last tick is exactly at vmax
    if ticks[-1] > vmax:
        ticks = ticks[:-1]
    # Add vmax if it's not already included
    if ticks[-1] < vmax:
        ticks = np.append(ticks, vmax)

    positions = [positions_text_labels[pos] for pos in positions]

    # Create heatmap with mask and improved styling
    heatmap = sns.heatmap(
        agreement_matrix,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        xticklabels=positions,
        yticklabels=positions,
        cbar_kws={
            "label": "Cohen's Kappa",
            "pad": 0.02,
            "ticks": ticks,
            "format": "%.2f",  # Show 2 decimal places
        },
        annot_kws={"size": annot_size, "weight": "bold"},
    )

    heatmap.figure.axes[-1].yaxis.label.set_size(tick_size)
    heatmap.figure.axes[-1].tick_params(labelsize=tick_size - 5)

    # Improve axis labels and ticks
    plt.xticks(rotation=0, ha="center")
    plt.yticks(rotation=0, va="center")
    ax.tick_params(axis="both", which="major", labelsize=tick_size)

    # Add title if provided
    if title:
        plt.title(title, pad=20, size=tick_size + 2, weight="bold")

    # Add subtle grid lines
    ax.grid(False)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure with high quality settings
    plt.savefig(
        output_path,
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
        edgecolor="none",
        format=output_path.split(".")[-1],
    )
    plt.close()


def get_positions(task_config):
    # Determine position types based on classification path
    is_corners = "corners" in task_config["extreme_pos_path"]

    if is_corners:
        positions = ["top_left", "top_right", "bottom_left", "bottom_right"]
    else:
        positions = ["leftmost", "rightmost"]

    return positions


def safe_cohen_kappa(y1, y2, label_type):
    # If both arrays contain only one unique value and they're equal
    if len(set(y1)) == 1 and len(set(y2)) == 1 and set(y1) == set(y2):
        return 1.0

    try:
        return cohen_kappa_score(
            y1, y2
        )
    except Exception as e:
        print(f"Error computing kappa for {label_type}: {str(e)}")
        return np.nan


def compute_inter_agreement(df, pos1, pos2, pairwise_results):
    print(f"\nComputing inter-agreement between {pos1} and {pos2}")
    labels = list(df.columns[4:])

    # Pre-filter dataframes for efficiency
    df1 = df[df["persona_pos"] == pos1]
    df2 = df[df["persona_pos"] == pos2]

    if df1.empty or df2.empty:
        print(f"Warning: No data for position {pos1} or {pos2}")
        return np.nan

    personas1 = df1["persona_id"].unique()
    personas2 = df2["persona_id"].unique()
    kappas = []

    for label in labels:
        print(f"Processing label: {label}")
        persona_kappas = []

        # Compare each persona from pos1 with each persona from pos2
        for p1 in personas1:
            for p2 in personas2:
                # Get all predictions for each persona
                p1_preds = df1[df1["persona_id"] == p1][label].values
                p2_preds = df2[df2["persona_id"] == p2][label].values

                kappa = safe_cohen_kappa(p1_preds, p2_preds, label)
                if not np.isnan(kappa):
                    persona_kappas.append(kappa)
                    pairwise_results.append((p1, p2, kappa))

        # Average agreement for this label
        if persona_kappas:
            mean_kappa = np.mean(persona_kappas)
            kappas.append(mean_kappa)
            print(f"Average kappa for {label}: {mean_kappa:.3f}")

    return np.mean(kappas) if kappas else np.nan


def compute_intra_agreement(df, pos, pairwise_results):
    print(f"\nComputing intra-agreement for {pos}")
    labels = list(df.columns[4:])

    # Pre-filter dataframe for the given position
    pos_df = df[df["persona_pos"] == pos]

    if pos_df.empty:
        print(f"Warning: No data for position {pos}")
        return np.nan

    personas = pos_df["persona_id"].unique()
    kappas = []

    # create a list of all the labels that are in the df

    for label in labels:
        print(f"Processing label: {label}")
        persona_kappas = []

        # Compute agreement between all pairs of personas
        for i in range(len(personas)):
            for j in range(i + 1, len(personas)):
                p1, p2 = personas[i], personas[j]

                # Get all predictions for each persona
                p1_preds = pos_df[pos_df["persona_id"] == p1][label].values
                p2_preds = pos_df[pos_df["persona_id"] == p2][label].values

                kappa = safe_cohen_kappa(p1_preds, p2_preds, label)
                if not np.isnan(kappa):
                    persona_kappas.append(kappa)
                    pairwise_results.append((p1, p2, kappa))

        # Average agreement for this label
        if persona_kappas:
            mean_kappa = np.mean(persona_kappas)
            kappas.append(mean_kappa)
            print(f"Average kappa for {label}: {mean_kappa:.3f}")

    return np.mean(kappas) if kappas else np.nan

def compute_inter_agreement_optimized(pos1, pos2, labels, position_data, persona_predictions, pairwise_results):
    """Optimized inter-agreement computation using pre-computed data"""
    print(f"\nComputing inter-agreement between {pos1} and {pos2}")
    
    # Quick check for empty data
    if not position_data[pos1]['personas'].size or not position_data[pos2]['personas'].size:
        print(f"Warning: No data for position {pos1} or {pos2}")
        return np.nan
    
    personas1 = position_data[pos1]['personas']
    personas2 = position_data[pos2]['personas']
    
    kappas = []
    
    for label in labels:
        print(f"Processing label: {label}")
        persona_kappas = []
        
        # Use pre-computed predictions - no DataFrame filtering in inner loop
        for p1 in personas1:
            p1_preds = persona_predictions[(pos1, p1)][label]
            
            for p2 in personas2:
                p2_preds = persona_predictions[(pos2, p2)][label]
                
                kappa = safe_cohen_kappa(p1_preds, p2_preds, label)
                if not np.isnan(kappa):
                    persona_kappas.append(kappa)
                pairwise_results.append((p1, p2, kappa))
        
        if persona_kappas:
            mean_kappa = np.mean(persona_kappas)
            kappas.append(mean_kappa)
            print(f"Average kappa for {label}: {mean_kappa:.3f}")
    
    print(f"Done processing label: {label} for inter-agreement positions {pos1} and {pos2}")
    return np.mean(kappas) if kappas else np.nan

def compute_intra_agreement_optimized(pos, labels, position_data, persona_predictions, pairwise_results):
    """Optimized intra-agreement computation using pre-computed data"""
    print(f"\nComputing intra-agreement for {pos}")
    
    # Quick check for empty data
    if not position_data[pos]['personas'].size:
        print(f"Warning: No data for position {pos}")
        return np.nan
    
    personas = position_data[pos]['personas']
    kappas = []
    
    for label in labels:
        print(f"Processing label: {label}")
        persona_kappas = []
        
        # Use pre-computed predictions - no DataFrame filtering in inner loop
        for i in range(len(personas)):
            p1 = personas[i]
            p1_preds = persona_predictions[(pos, p1)][label]
            
            for j in range(i + 1, len(personas)):
                p2 = personas[j]
                p2_preds = persona_predictions[(pos, p2)][label]
                
                kappa = safe_cohen_kappa(p1_preds, p2_preds, label)
                if not np.isnan(kappa):
                    persona_kappas.append(kappa)
                pairwise_results.append((p1, p2, kappa))
        
        if persona_kappas:
            mean_kappa = np.mean(persona_kappas)
            kappas.append(mean_kappa)
            print(f"Average kappa for {label}: {mean_kappa:.3f}")
    
    print(f"Done processing label: {label} for intra-agreement position {pos}")
    return np.mean(kappas) if kappas else np.nan

# Wrapper functions for multiprocessing (must be picklable)
def compute_agreement_task(args):
    """Worker function that computes agreement for a single position pair"""
    i, j, pos1, pos2, labels, position_data, persona_predictions = args
    
    print(f"\nProcessing position pair ({pos1}, {pos2})")
    
    if i == j:
        result = compute_intra_agreement_optimized(
            pos1, labels, position_data, persona_predictions, []
        )
    else:
        result = compute_inter_agreement_optimized(
            pos1, pos2, labels, position_data, persona_predictions, []
        )
    
    return (i, j, result)

def main():
    global DATASET_NAME

    parser = argparse.ArgumentParser(description="Run content classification pipeline")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/results/text_classification/Qwen2.5-32B-Instruct/20250722_125943/final_results.json"
    )
    args = parser.parse_args()

    try:
        with open(args.input_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return
    
    #task_config, model_config, data_df = read_results()

    task_config = data["metadata"]["task_config"]
    model_config = data["metadata"]["model_config"]

    MODEL_NAME = model_config["name"].split("/")[-1]
    TIMESTAMP = args.input_path.split("/")[-2] # or [-1] in the old version
    DATASET_NAME = data["metadata"]["dataset_name"]

    first_result_labels = data["results"][0]["true_labels"]
    # extract the keys of the labels:
    first_result_labels = list(first_result_labels.keys())
    # label names
    is_harmful_label = first_result_labels[0]
    target_group_label = first_result_labels[1] if len(first_result_labels) > 1 else None
    attack_method_label = first_result_labels[2] if len(first_result_labels) > 2 else None
    
    #target_group_label = "target_category"  # "target_group" # target_category
    #attack_method_label = "attack_method"  # "attack_method", None
    

    plot_path = f"images/agreement/{MODEL_NAME}/agreement_matrix_{TIMESTAMP}.png"
    pairwise_path = f"images/agreement/{MODEL_NAME}/pairwise_agreements_{TIMESTAMP}.csv"

    predictions = data["results"]
    df = pd.DataFrame(predictions)
    print(df.columns)
    

    print(df.columns)
    print()
    print("=" * 70)
    print(f"\nDataset size: {len(df)} rows")
    print(f"Unique images: {df['item_id'].nunique()}")
    print(f"Positions present: {df['persona_pos'].unique()}")
    print(f"Personas present: {df['persona_id'].nunique()}")
    print("=" * 70)
    print()

    # Extract is_hate_speech
    df["is_hate_speech"] = df["predicted_labels"].apply(lambda x: int(x[is_harmful_label]))

    # Extract all other columns automatically
    if len(df["predicted_labels"].iloc[0]) > 1:  # Check if there are additional columns
        sample_labels = df["predicted_labels"].iloc[0]
        other_columns = [col for col in sample_labels.keys() if col != is_harmful_label]
        
        for col in other_columns:
            df[col] = df["predicted_labels"].apply(lambda x: x[col])

    df = df.drop(["predicted_labels", "true_labels"], axis=1)

    # Define positions
    positions = get_positions(task_config)
    matrix_size = len(positions)

    # Create agreement matrix
    agreement_matrix = np.zeros((matrix_size, matrix_size))
    pairwise_results = []
    labels, position_data, persona_predictions = precompute_data_structures(df, positions)

    tasks = []
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions[i:], i):
            tasks.append((i, j, pos1, pos2, labels, position_data, persona_predictions))

    num_processes = min(mp.cpu_count() - 1, len(tasks))
    print(f"Number of available processors: {mp.cpu_count()}")
    print(f"Running {len(tasks)} tasks on {num_processes} processes")
    
    # Create process pool and run tasks
    with mp.Pool(num_processes) as pool:
        results = pool.map(compute_agreement_task, tasks)
    
    # Fill the agreement matrix with results
    for i, j, result in results:
        agreement_matrix[i, j] = result
        if i != j:
            agreement_matrix[j, i] = result

    # Create DataFrame for better formatting
    agreement_df = pd.DataFrame(agreement_matrix, index=positions, columns=positions)

    # Get the directory name from the path
    output_dir = os.path.dirname(plot_path)
    # Create the directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    plot_agreement_matrix(agreement_df, positions, plot_path)
    save_pairwise_comparisons(pairwise_results, pairwise_path)

    # Print formatted matrix
    print("\nAgreement Matrix:")
    print("-" * 65)
    print(agreement_df.round(3))


if __name__ == "__main__":
    main()
