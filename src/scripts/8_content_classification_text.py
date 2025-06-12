import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

from src.models.base import ModelConfig
from src.datasets.base import (
    BaseDataset,
    PredictionParser,
    LabelConverter,
    DatasetConfig,
)
from src.datasets.hate_speech_text_dataset import HateSpeechTextDataset, text_collate_fn
from src.datasets.hate_speech_text_parser import (
    HateSpeechJsonParser,
    HateSpeechLabelConverter,
)
from src.models.vllm_model import VLLMModel
from utils.util import get_gpu_memory_info


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Structured result for a single classification."""

    item_id: str
    true_labels: Dict[str, Any]
    raw_prediction: str
    predicted_labels: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== Evaluation Framework ====================


class ClassificationEvaluator:
    """Handles evaluation and metrics calculation."""

    def __init__(self, aspects: List[str]):
        self.aspects = aspects

    def calculate_metrics(
        self, results: List[ClassificationResult]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each aspect."""
        metrics = {}

        for aspect in self.aspects:
            y_true = [self._get_aspect_value(r.true_labels, aspect) for r in results]
            y_pred = [
                self._get_aspect_value(r.predicted_labels, aspect) for r in results
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


# ==================== Text Pipeline ====================


class TextClassificationPipeline:
    """Pipeline for running text classification experiments."""

    def __init__(
        self,
        dataset: BaseDataset,
        model: VLLMModel,
        parser: PredictionParser,
        converter: LabelConverter,
        evaluator: ClassificationEvaluator,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self.dataset = dataset
        self.model = model
        self.parser = parser
        self.converter = converter
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.num_workers = num_workers

    def run(self) -> Tuple[List[ClassificationResult], Dict[str, Dict[str, float]]]:
        """Run the classification pipeline."""
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=text_collate_fn,
        )

        results = []

        logger.info(
            f"Processing {len(self.dataset)} items in batches of {self.batch_size}..."
        )

        for batch_data in tqdm(dataloader):
            inputs_list, labels_list, ids_list, metadata_list = batch_data
            
            # Extract prompts from inputs
            prompts = [inp["prompt"] for inp in inputs_list]
            
            # Process batch through VLLM
            predictions = self.model.process_batch(prompts)

            # Process each prediction
            for idx, pred in enumerate(predictions):
                true_labels = self.converter.convert(labels_list[idx])
                predicted_labels = self.parser.parse(pred)

                result = ClassificationResult(
                    item_id=ids_list[idx],
                    true_labels=true_labels,
                    raw_prediction=pred,
                    predicted_labels=predicted_labels,
                    metadata=metadata_list[idx] if metadata_list else {},
                )

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
    parser = argparse.ArgumentParser(description="Text-based Classification Framework")

    # Dataset arguments
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="hate_speech_text",
        choices=["hate_speech_text", "custom"],
        help="Type of dataset to use",
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset CSV file"
    )
    parser.add_argument(
        "--prompts_file", type=str, default=None, help="Path to prompts parquet file (optional)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="vllm",
        choices=["vllm"],
        help="Type of model to use",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model ID from HuggingFace",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for text generation",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate",
    )

    # Pipeline arguments
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for DataLoader"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/text_classification_results.json",
        help="Path to save results",
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Log configuration
    logger.info("Starting text classification pipeline...")
    logger.info(f"Dataset: {args.dataset_type}")
    logger.info(f"Model: {args.model_type} ({args.model_id})")
    logger.info(f"GPU State: {get_gpu_memory_info()}")

    # Initialize model
    model_config = ModelConfig(
        model_id=args.model_id,
        additional_params={
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": 22,
        },
    )

    if args.model_type == "vllm":
        model = VLLMModel(model_config)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Initialize dataset components based on dataset type
    dataset_config = DatasetConfig(
        data_path=args.data_path, max_samples=args.max_samples
    )

    if args.dataset_type == "hate_speech_text":
        # Get processor/tokenizer from model
        processor = model.tokenizer
        
        dataset = HateSpeechTextDataset(
            dataset_config, processor, args.prompts_file
        )
        parser = HateSpeechJsonParser()
        converter = HateSpeechLabelConverter()
        evaluator = ClassificationEvaluator(
            ["is_hate_speech", "target_category", "specific_target"]
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    # Create and run pipeline
    pipeline = TextClassificationPipeline(
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

    # Cleanup model resources
    model.cleanup()

    logger.info("Pipeline completed successfully!")


# ==================== Utility Functions ====================


def save_results(
    results: List[ClassificationResult],
    metrics: Dict[str, Dict[str, float]],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save results and metrics to file."""
    output_data = {
        "results": [
            {
                "item_id": r.item_id,
                "true_labels": r.true_labels,
                "predicted_labels": r.predicted_labels,
                "raw_prediction": r.raw_prediction,
                "metadata": r.metadata,
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


if __name__ == "__main__":
    main()