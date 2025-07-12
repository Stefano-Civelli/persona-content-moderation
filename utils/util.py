import json
from typing import List, Dict, Any, Optional
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
from pathlib import Path
from sklearn.metrics import classification_report
import logging
import yaml

logger = logging.getLogger(__name__)


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
        ],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


class ClassificationEvaluator:
    """Handles evaluation and metrics calculation."""

    def __init__(self, aspects: List[str]):
        self.aspects = aspects

    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        if not results:
            return {}

        metrics = {}

        filtered_true_labels = [
            {k: v for k, v in r["true_labels"].items() if k in self.aspects}
            for r in results
        ]
        filtered_pred_labels = [
            {k: v for k, v in r["predicted_labels"].items() if k in self.aspects}
            for r in results
        ]

        y_true_combined = [tuple(labels.values()) for labels in filtered_true_labels]
        y_pred_combined = [tuple(labels.values()) for labels in filtered_pred_labels]

        correct_count = sum(
            1 for true, pred in zip(y_true_combined, y_pred_combined) if true == pred
        )
        metrics["overall"] = {"exact_match_ratio": correct_count / len(results)}

        for aspect in self.aspects:
            y_true = [
                self._get_aspect_value(labels, aspect)
                for labels in filtered_true_labels
            ]
            y_pred = [
                self._get_aspect_value(labels, aspect)
                for labels in filtered_pred_labels
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


# ================ YAML UTILS ================


def load_config(config_path="models_config.yaml", prompts_path="prompts.yaml"):
    """Load the main configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(prompts_path, "r") as f:
        prompts = yaml.safe_load(f)
    return config, prompts


def get_task_config(config: Dict[str, Any], task_type: str):
    task_key = f"{task_type}_config"
    if task_key in config:
        return config[task_key]
    return None


def get_model_config(task_config: Dict[str, Any], model_name: str):
    cleaned_task_config = {k: v for k, v in task_config.items() if k != "models"}
    for model_conf in task_config.get("models", []):
        if model_conf["name"] == model_name:
            return model_conf, cleaned_task_config
    return None, None


def get_all_model_names(config_path: str = "models_config.yaml") -> List[str]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    vision_model_names = [
        model["name"] for model in config.get("vision_config", {}).get("models", [])
    ]
    text_model_names = [
        model["name"] for model in config.get("text_config", {}).get("models", [])
    ]
    return vision_model_names + text_model_names


# ================ YAML UTILS ================


def clean_leading_trailing_newline(s: str) -> str:
    """Remove leading and trailing newlines from a string."""
    return s.strip("\n") if isinstance(s, str) else s
