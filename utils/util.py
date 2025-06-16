import torch
import json
from typing import List, Dict, Any, Optional
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
from pathlib import Path
from sklearn.metrics import classification_report


def get_gpu_memory_info() -> str:
    """Get GPU memory information."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
    return "No GPU available"


# def save_results(
#     results: List[Dict[str, Any]],
#     metadata: Dict[str, Any] = None,
#     output_path: str = "../results",
# ) -> None:
#     """Save prediction results to a JSON file with timestamp"""
#     os.makedirs(output_path, exist_ok=True)
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_file = os.path.join(output_path, f"predictions_{timestamp}.json")

#     output_data = {
#         "metadata": {"timestamp": timestamp, **(metadata or {})},
#         "predictions": results,
#     }

#     with open(output_file, "w") as f:
#         json.dump(output_data, f, indent=2)

#     print(f"\nResults saved to: {output_file}")


def save_results(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Dict[str, float]],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save results and metrics to file."""
    output_data = {
        "metrics": metrics,
        "metadata": metadata or {},
        "results": [
            {
                "item_id": r["item_id"],
                "true_labels": r["true_labels"],
                "predicted_labels": r["predicted_labels"],
                "raw_prediction": r["raw_prediction"],
                "persona_id": r["persona_id"],
                "persona_pos": r["persona_pos"],
            }
            for r in results
        ]
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


class ClassificationEvaluator:
    """Handles evaluation and metrics calculation."""

    def __init__(self, aspects: List[str]):
        self.aspects = aspects

    def calculate_metrics(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each aspect."""
        metrics = {}

        for aspect in self.aspects:
            y_true = [self._get_aspect_value(r["true_labels"], aspect) for r in results]
            y_pred = [
                self._get_aspect_value(r["predicted_labels"], aspect) for r in results
            ]

            report = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )

            metrics[aspect] = {
                "accuracy": report["accuracy"],
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_f1": report["weighted avg"]["f1-score"],
            }

        return metrics

    def _get_aspect_value(self, labels: Dict, aspect: str) -> str:
        """Extract aspect value from labels."""
        value = labels.get(aspect, "unknown")
        return str(value).lower()


def visualize_agreement_matrix(
    results_dict,
    output_path="agreement_matrix.pdf",
    figsize=(10, 4),
    cmap="YlOrRd",
    vmin=None,
    vmax=None,
    tick_size=12,
    annot_size=11,
    title=None,
):
    # Extract the agreement matrix
    agreement_matrix = results_dict["combined"]["agreement_matrix"]

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(agreement_matrix), k=1)

    # Set up the positions labels
    positions = ["TR", "TL", "BR", "BL"]

    # Create figure and axis with higher DPI
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # If vmin/vmax not provided, set them based on data
    if vmin is None:
        vmin = np.floor(np.min(agreement_matrix) * 100) / 100
    if vmax is None:
        vmax = np.ceil(np.max(agreement_matrix) * 100) / 100

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
            "ticks": plt.MaxNLocator(5),  # Limit number of ticks for cleaner look
        },
        annot_kws={"size": annot_size, "weight": "bold"},
    )

    heatmap.figure.axes[-1].yaxis.label.set_size(tick_size)
    heatmap.figure.axes[-1].tick_params(labelsize=tick_size)

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


if __name__ == "__main__":

    file_name = "predictions_20241220_115949"
    with open(f"../results/agreement_matrix_{file_name}.pkl", "rb") as f:
        results = pickle.load(f)

    visualize_agreement_matrix(
        results,
        output_path=f"../results/agreement_matrix_{file_name}.png",
        figsize=(8, 6),
        cmap="YlOrRd",
        tick_size=18,
        annot_size=22,
    )
