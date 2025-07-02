import json
import logging
import argparse
from typing import Dict, Any, List, Tuple
import yaml

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.vllm_model import VLLMModel

from src.models.base import BaseModel
from src.datasets.base import (
    BaseDataset,
    PredictionParser,

)
from src.models.vision_model import Idefics3Model
from utils.util import ClassificationEvaluator, save_results

import os

print(os.getcwd())

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        """
        Custom collate function for IDEFICS model with left padding

        Args:
            batch: List of tuples (inputs, labels, img_name)

        Returns:
            Tuple of (batched_inputs, batched_labels, img_names)
        """
        # Unzip the batch into separate lists
        inputs, labels, img_names, persona_id, persona_pos = zip(*batch)

        # Process inputs
        pixel_values = [item["pixel_values"] for item in inputs]
        pixel_attention_mask = [item["pixel_attention_mask"] for item in inputs]
        input_ids = [item["input_ids"] for item in inputs]
        attention_mask = [item["attention_mask"] for item in inputs]

        # print(f'Pixel values shape: {pixel_values[0].shape}')
        # Stack pixel values and pixel attention mask
        pixel_values = torch.stack(pixel_values)
        pixel_attention_mask = torch.stack(pixel_attention_mask)

        # Find max length in this batch
        max_len = max(ids.size(0) for ids in input_ids)
        batch_size = len(input_ids)

        # Create tensors for left-padded sequences
        padded_input_ids = torch.full(
            (batch_size, max_len), 128002, dtype=input_ids[0].dtype
        )
        padded_attention_mask = torch.zeros(
            (batch_size, max_len), dtype=attention_mask[0].dtype
        )

        # Fill in the sequences from the right
        for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
            seq_len = ids.size(0)
            padded_input_ids[i, -seq_len:] = ids
            padded_attention_mask[i, -seq_len:] = mask

        # Create batched inputs dictionary
        batched_inputs = {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
        }

        # Process labels - assuming they're dictionaries
        batched_labels = {}
        if labels and isinstance(labels[0], dict):
            all_keys = set().union(*[d.keys() for d in labels])

            for key in all_keys:
                values = [d.get(key) for d in labels]
                if all(isinstance(v, (int, float, bool, torch.Tensor)) for v in values):
                    if isinstance(values[0], torch.Tensor):
                        batched_labels[key] = torch.stack(values)
                    else:
                        batched_labels[key] = torch.tensor(values)
                else:
                    batched_labels[key] = values

        return (
            batched_inputs,
            batched_labels,
            list(img_names),
            list(persona_id),
            list(persona_pos),
        )

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

        for batch_inputs, batch_labels, item_ids, persona_ids, persona_pos in tqdm(
            dataloader
        ):
            predictions = self.model.process_batch(batch_inputs)

            for idx, pred in enumerate(predictions):
                true_labels = self.converter.convert(batch_labels)
                predicted_labels = self.parser.parse(pred)

                result = {
                    "item_id": item_ids[idx],
                    "true_labels": true_labels,
                    "raw_prediction": pred,
                    "predicted_labels": predicted_labels,
                    "persona_id": persona_ids[idx],
                    "persona_pos": persona_pos[idx],
                }

                results.append(result)

                # Log progress every 500 items
                if len(results) % 500 == 0:
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


def main():
    """Main execution function."""

    with open("config_text.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open("models.yaml", "r") as f:
        models = yaml.safe_load(f)

    MODEL = models["text_models"][config["model_id"]]

    config["prompts_file"] = (
        config["prompts_file"]
        .replace("[DATASET]", "YODER")
        .replace("[MODEL_NAME]", MODEL.split("/")[-1])
    )

    config["output_path"] = (
        config["output_path"]
        .replace("[MODEL_NAME]", MODEL.split("/")[-1])
        .replace("[DATETIME]", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(config["output_path"]), exist_ok=True)

    logger.info(f"Output path: {config['output_path']}")
    logger.info(f"Using model: {config['model_id']}")

    # ========== Create Objects ==========

    # model = Idefics3Model(
    #     model_id=config["model_id"],
    #     resolution_factor=config["resolution_factor"],
    #     max_new_tokens=config["max_new_tokens"]
    # )

    model = VLLMModel(
        model_id=MODEL,
        seed=config["vllm_seed"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_model_len=config["max_model_len"],
        max_num_seqs=config["max_num_seqs"],
        enforce_eager=config["enforce_eager"],
    )

    dataset = FacebookHatefulMemesDataset(
        config["data_path"],
        config["labels_relative_location"],
        model.get_processor(),
        config["prompts_file"],
        config["max_samples"],
    )

    # for predicted labels
    parser = HatefulMemesPredictionParser()
    # for true labels
    converter = HatefulMemesLabelConverter()
    evaluator = ClassificationEvaluator(["harmful", "target_group", "attack_method"])

    # Create and run pipeline
    pipeline = ClassificationPipeline(
        dataset=dataset,
        model=model,
        parser=parser,
        converter=converter,
        evaluator=evaluator,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    # ========== Create Objects ==========

    results, metrics = pipeline.run()

    logger.info("\nFinal Metrics:")
    pipeline._log_metrics(metrics)

    # Save results
    metadata = {
        "config": config,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    save_results(results, metrics, config["output_path"], metadata)
    logger.info(f"Results saved to {config['output_path']}")

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
