import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Protocol
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

from src.models.base import BaseModel
from src.datasets.base import (
    BaseDataset,
    PredictionParser,
    LabelConverter,
)
from src.datasets.facebook_hateful_memes_dataset import FacebookHatefulMemesDataset
from src.datasets.facebook_hateful_memes_parser import (
    HatefulMemesPredictionParser,
    HatefulMemesLabelConverter,
)
from src.models.idefics import Idefics3Model
from utils.util import get_gpu_memory_info

import os
print(os.getcwd())

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==================== Evaluation Framework ====================


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


# ==================== Main Pipeline ====================


class ClassificationPipeline:
    """Main pipeline for running classification experiments."""

    def __init__(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        parser: PredictionParser,
        converter: LabelConverter,
        evaluator: ClassificationEvaluator,
        batch_size: int = 2,
        num_workers: int = 4,
    ):
        self.dataset = dataset
        self.model = model
        self.parser = parser
        self.converter = converter
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.num_workers = num_workers

    def custom_collate_fn(self, batch):
        """Custom collate function for DataLoader."""
        inputs_list, labels_list, ids_list, metadata_list = zip(*batch)

        # Stack inputs
        stacked_inputs = {}
        for key in inputs_list[0].keys():
            stacked_inputs[key] = torch.stack([inp[key] for inp in inputs_list])

        # For now, return first label set (assuming single-sample batches for multi-label)
        return stacked_inputs, labels_list[0], ids_list, metadata_list

    def run(self) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
        """Run the classification pipeline."""
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.custom_collate_fn,
        )

        results = []

        logger.info(
            f"Processing {len(self.dataset)} items in batches of {self.batch_size}..."
        )

        for batch_inputs, batch_labels, item_ids, metadata_list in tqdm(dataloader):
            predictions = self.model.process_batch(batch_inputs)

            for idx, pred in enumerate(predictions):
                true_labels = self.converter.convert(batch_labels)
                predicted_labels = self.parser.parse(pred)

                result = {
                    "item_id": item_ids[idx],
                    "true_labels": true_labels,
                    "raw_prediction": pred,
                    "predicted_labels": predicted_labels,
                    "metadata": metadata_list[idx] if metadata_list else {},
                }

                results.append(result)

                # Log progress every 100 items
                if len(results) % 100 == 0:
                    running_metrics = self.evaluator.calculate_metrics(results)
                    logger.info(f"\nProgress: {len(results)} items processed")
                    self._log_metrics(running_metrics)

        # Calculate final metrics
        final_metrics = self.evaluator.calculate_metrics(results)

        return results, final_metrics

    def _log_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """Log metrics to console."""
        for aspect, aspect_metrics in metrics.items():
            logger.info(f"\n{aspect.upper()} metrics:")
            for metric_name, value in aspect_metrics.items():
                logger.info(f"  {metric_name}: {value:.3f}")


# ==================== Main Entry Point ====================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-label Classification Framework")

    # Dataset arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/facebook-hateful-memes",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="data/processed/img_classification_prompts.pqt",
        help="Path to prompts file (optional)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/results/classification/[MODEL_NAME]/[DATETIME].json",
        help="Path to save results",
    )

    # Model arguments
    parser.add_argument(
        "--model_id",
        type=str,
        default="HuggingFaceM4/Idefics3-8B-Llama3",
        help="Model ID from HuggingFace",
    )
    parser.add_argument(
        "--resolution_factor",
        type=int,
        default=4,
        help="Resolution factor for image processing",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate",
    )

    # Pipeline arguments
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for inference"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for DataLoader"
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Log configuration
    logger.info("Starting classification pipeline...")
    logger.info(f"GPU State: {get_gpu_memory_info()}")

    args.output_path = args.output_path.replace(
        "[MODEL_NAME]", args.model_id.split("/")[-1]
    ).replace("[DATETIME]", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))

    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Using model: {args.model_id}")

    model = Idefics3Model(
        model_id=args.model_id,
        additional_params={
            "resolution_factor": args.resolution_factor,
            "max_new_tokens": args.max_new_tokens,
        },
    )

    dataset = FacebookHatefulMemesDataset(
        args.data_path,
        model.get_processor(),
        args.prompts_file,
        args.max_samples,
    )
    parser = HatefulMemesPredictionParser()
    converter = HatefulMemesLabelConverter()
    evaluator = ClassificationEvaluator(["harmful", "target_group", "attack_method"])

    # Create and run pipeline
    pipeline = ClassificationPipeline(
        dataset=dataset,
        model=model,
        parser=parser,
        converter=converter,
        evaluator=evaluator,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    results, metrics = pipeline.run()

    # Log final metrics
    logger.info("\nFinal Metrics:")
    pipeline._log_metrics(metrics)

    # Save results
    metadata = {
        "args": vars(args),
        "device": model.config.device,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    save_results(results, metrics, args.output_path, metadata)

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()


# ==================== Utility Functions ====================


def save_results(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Dict[str, float]],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save results and metrics to file."""
    output_data = {
        "results": [
            {
                "item_id": r["item_id"],
                "true_labels": r["true_labels"],
                "predicted_labels": r["predicted_labels"],
                "raw_prediction": r["raw_prediction"],
                "metadata": r["metadata"],
            }
            for r in results
        ],
        "metrics": metrics,
        "metadata": metadata or {},
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to {output_path}")
