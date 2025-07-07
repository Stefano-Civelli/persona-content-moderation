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

DATASET_NAME = None

# Define possible values for each label
POSSIBLE_VALUES = {
    "facebook-hateful-memes": {
        "is_hate_speech": [0, 1],
        "target_group": [
            "none",
            "disability",
            "race",
            "religion",
            "nationality",
            "sex",
        ],
        "attack_method": [
            "none",
            "contempt",
            "mocking",
            "inferiority",
            "slurs",
            "exclusion",
            "dehumanizing",
            "inciting_violence",
        ],
    },
    "MMHS150K": {
        "is_hate_speech": [0, 1],
        "target_group": ["Racist", "Sexist", "Homophobe", "Religion", "OtherHate"],
    },
    "yoder": {
        "is_hate_speech": [0, 1],
        "target_group": [
            "women",
            "black",
            "lgbtq+",
            "muslims/arabic",
            "asian",
            "latino/hispanic",
            "jews",
            "white",
            "men",
            "christians",
            "none",
        ],
    },
}

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
            y1, y2, labels=POSSIBLE_VALUES[DATASET_NAME][label_type]
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


def main():
    global DATASET_NAME
    parser = argparse.ArgumentParser(description="Run content classification pipeline")
    parser.add_argument("--task_type", type=str, default="text", help="Type of task") # text, img
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-32B-Instruct",
        help="Model name/path",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default="20250704_183607" # 20250625_162048, 20250703_150720
    )
    parser.add_argument(
        "--extreme_personas_type",
        type=str,
        default="extreme_pos_corners_100",  # extreme_pos_left_right
    )
    args = parser.parse_args()

    # label names
    is_harmful_label = "is_hate_speech"
    target_group_label = "target_category"  # "target_group" # target_category
    #attack_method_label = "attack_method"  # "attack_method", None
    
    MODEL_NAME = args.model.split("/")[-1]

    input_path = f"data/results/{args.task_type}_classification/{MODEL_NAME}/{args.timestamp}.json"
    # New implementation with the folder:
    #input_path_folder = f"data/results/{args.task_type}_classification/{MODEL_NAME}/{args.timestamp}/final_results.json"
    plot_path = f"images/agreement/{MODEL_NAME}/agreement_matrix_{args.timestamp}.png"
    pairwise_path = f"images/agreement/{MODEL_NAME}/pairwise_agreements_{args.timestamp}.csv"

    try:
        with open(input_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return

    task_config = data["metadata"]["task_config"]
    model_config = data["metadata"]["model_config"]

    assert model_config["name"].split('/')[-1] == MODEL_NAME, f"Model name mismatch: {model_config['name']} != {MODEL_NAME}"

    DATASET_NAME = task_config["dataset_name"].replace("_", "-")
    

    predictions = data["results"]
    df = pd.DataFrame(predictions)

    print(df.columns)
    print()
    print("=" * 70)
    print(f"\nDataset size: {len(df)} rows")
    print(f"Unique images: {df['item_id'].nunique()}")
    print(f"Positions present: {df['persona_pos'].unique()}")
    print(f"Personas present: {df['persona_id'].nunique()}")
    print("=" * 70)
    print()

    # Extract predicted labels
    df["is_hate_speech"] = df["predicted_labels"].apply(lambda x: int(x[is_harmful_label]))
    if DATASET_NAME == "facebook-hateful-memes":
        df["target_group"] = df["predicted_labels"].apply(lambda x: x["target_group"])
        df["attack_method"] = df["predicted_labels"].apply(lambda x: x["attack_method"])
    elif DATASET_NAME == "MMHS150K":
        df["target_group"] = df["predicted_labels"].apply(lambda x: x["hate_type"])
    elif DATASET_NAME == "yoder":
        df["target_group"] = df["predicted_labels"].apply(lambda x: x[target_group_label])
    else:
        print("Error: Dataset not supported")
        return

    df = df.drop(["predicted_labels", "true_labels"], axis=1)

    # Define positions
    positions = get_positions(task_config)

    matrix_size = len(positions)

    # Create agreement matrix
    agreement_matrix = np.zeros((matrix_size, matrix_size))
    pairwise_results = []

    # Fill matrix
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions[i:], i):
            print(f"\nProcessing position pair ({pos1}, {pos2})")
            if i == j:
                agreement_matrix[i, j] = compute_intra_agreement(
                    df, pos1, pairwise_results
                )
            else:
                # Compute for upper triangle
                agreement_matrix[i, j] = compute_inter_agreement(
                    df, pos1, pos2, pairwise_results
                )
                # Fill in lower triangle using symmetry
                agreement_matrix[j, i] = agreement_matrix[i, j]

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
