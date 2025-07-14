import argparse
import json
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import simpledorff
import os

# NOTE ON THE CHANGE OF METRIC:
# This script has been modified to use Krippendorff's Alpha instead of Cohen's Kappa.
# Krippendorff's Alpha is a more robust metric for calculating agreement between
# multiple annotators (in this case, personas).
# As a result, the concept of "pairwise" agreement between individual personas is
# no longer calculated. Instead, we compute a single agreement score for an entire
# group (for intra-agreement) or a combined set of groups (for inter-agreement).
# Consequently, the function to save pairwise results has been removed.

DATASET_NAME = None

# This dictionary is not strictly needed for Krippendorff's Alpha as it can infer
# the labels, but it's kept in case of future use.
POSSIBLE_VALUES = {
    "facebook-hateful-memes": {
        "harmful": [0, 1],
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
        "harmful": [0, 1],
        "target_group": ["Racist", "Sexist", "Homophobe", "Religion", "OtherHate"],
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

# The function to save pairwise comparisons is no longer needed with Krippendorff's Alpha.
# def save_pairwise_comparisons(pairwise_results, output_path): ...


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
    mask = np.triu(np.ones_like(agreement_matrix), k=1)
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    if vmin is None:
        vmin = (
            np.floor(np.nanmin(agreement_matrix) * 100) / 100
            if not np.all(np.isnan(agreement_matrix))
            else 0
        )
    if vmax is None:
        vmax = (
            np.ceil(np.nanmax(agreement_matrix) * 100) / 100
            if not np.all(np.isnan(agreement_matrix))
            else 1
        )

    ticks = np.arange(vmin, vmax + 0.02, 0.02)
    if len(ticks) > 0 and ticks[-1] > vmax:
        ticks = ticks[:-1]
    if len(ticks) == 0 or ticks[-1] < vmax:
        ticks = np.append(ticks, vmax)

    positions_labels = [positions_text_labels[pos] for pos in positions]

    heatmap = sns.heatmap(
        agreement_matrix,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        xticklabels=positions_labels,
        yticklabels=positions_labels,
        cbar_kws={
            # <-- The label is updated here
            "label": "Krippendorff's Alpha",
            "pad": 0.02,
            "ticks": ticks,
            "format": "%.2f",
        },
        annot_kws={"size": annot_size, "weight": "bold"},
    )

    heatmap.figure.axes[-1].yaxis.label.set_size(tick_size)
    heatmap.figure.axes[-1].tick_params(labelsize=tick_size - 5)
    plt.xticks(rotation=0, ha="center")
    plt.yticks(rotation=0, va="center")
    ax.tick_params(axis="both", which="major", labelsize=tick_size)

    if title:
        plt.title(title, pad=20, size=tick_size + 2, weight="bold")

    ax.grid(False)
    plt.tight_layout()
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




def compute_inter_agreement(df, pos1, pos2):
    print(f"\nComputing inter-agreement between {pos1} and {pos2}")

    # Identify which label columns are present in the dataframe
    label_cols = [
        col for col in ["harmful", "target_group", "attack_method"] if col in df.columns
    ]

    # Filter for the two groups you want to compare
    combined_df = df[df["persona_pos"].isin([pos1, pos2])]

    # Krippendorff's Alpha requires at least 2 annotators.
    if combined_df["persona_id"].nunique() < 2:
        print(
            f"Warning: Not enough unique annotators for positions {pos1} and {pos2}. Skipping."
        )
        return np.nan

    alphas = []
    for label in label_cols:
        print(f"Processing label: {label}")
        # Calculate Krippendorff's Alpha for the combined group on a single label
        alpha_score = simpledorff.calculate_krippendorffs_alpha_for_df(
            combined_df,
            experiment_col="item_id",
            annotator_col="persona_id",
            class_col=label,
        )
        if not np.isnan(alpha_score):
            alphas.append(alpha_score)
            print(f"Krippendorff's Alpha for {label}: {alpha_score:.3f}")

    # Return the average Alpha across all labels
    return np.mean(alphas) if alphas else np.nan


def compute_intra_agreement(df, pos):
    print(f"\nComputing intra-agreement for {pos}")

    label_cols = [
        col for col in ["harmful", "target_group", "attack_method"] if col in df.columns
    ]

    # Filter for the specific group
    group_df = df[df["persona_pos"] == pos]

    if group_df.empty or group_df["persona_id"].nunique() < 2:
        print(f"Warning: Not enough data or annotators for position {pos}. Skipping.")
        return np.nan

    alphas = []
    for label in label_cols:
        print(f"Processing label: {label}")
        # Calculate Krippendorff's Alpha for the group on a single label
        alpha_score = simpledorff.calculate_krippendorffs_alpha_for_df(
            group_df,
            experiment_col="item_id",
            annotator_col="persona_id",
            class_col=label,
        )
        if not np.isnan(alpha_score):
            alphas.append(alpha_score)
            print(f"Krippendorff's Alpha for {label}: {alpha_score:.3f}")

    # Return the average Alpha across all labels
    return np.mean(alphas) if alphas else np.nan


def main():
    global DATASET_NAME

    parser = argparse.ArgumentParser(description="Run content classification pipeline")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/results/img_classification/Qwen2.5-VL-7B-Instruct/20250712_012750.json/final_results.json"
    )
    args = parser.parse_args()

    try:
        with open(args.input_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return
    
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

    plot_path = f"images/agreement/{MODEL_NAME}/agreement_matrix_{TIMESTAMP}.png"
    pairwise_path = f"images/agreement/{MODEL_NAME}/pairwise_agreements_{TIMESTAMP}.csv"

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
    if DATASET_NAME == "facebook":
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

    positions = get_positions(task_config)
    matrix_size = len(positions)
    agreement_matrix = np.zeros((matrix_size, matrix_size))

    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions[i:], i):
            print(f"\nProcessing position pair ({pos1}, {pos2})")
            if i == j:
                # Pass only the dataframe and position to the simplified function
                agreement_matrix[i, j] = compute_intra_agreement(df, pos1)
            else:
                # Pass only the dataframe and two positions
                agreement_matrix[i, j] = compute_inter_agreement(df, pos1, pos2)
                agreement_matrix[j, i] = agreement_matrix[i, j]

    agreement_df = pd.DataFrame(agreement_matrix, index=positions, columns=positions)

    output_dir = os.path.dirname(plot_path)
    os.makedirs(output_dir, exist_ok=True)

    plot_agreement_matrix(agreement_df, positions, plot_path)
    print(f"\nAgreement matrix plot saved to {plot_path}")

    # The pairwise CSV saving is removed as it's not applicable to Krippendorff's Alpha.

    print("\nAgreement Matrix (Krippendorff's Alpha):")
    print("-" * 65)
    print(agreement_df.round(3))


if __name__ == "__main__":
    main()
