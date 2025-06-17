import json
import logging
import argparse
from typing import Dict, Any, List, Tuple
import yaml

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.yoder_text_parser import YoderLabelConverter, YoderPredictionParser
from src.models.base import (
    BaseModel as ScriptBaseModel,
)  # Renamed to avoid conflict with pydantic.BaseModel
from src.models.vllm_model import VLLMModel
from src.datasets.base import (
    BaseDataset,
    PredictionParser,
    LabelConverter,
)

# Assuming YoderIdentityDataset and IdentityContentClassification are in yoder_text_dataset
from src.datasets.yoder_text_dataset import (
    YoderIdentityDataset,
    map_grouping,  # If needed by converter/parser
)
from utils.util import ClassificationEvaluator, get_gpu_memory_info, save_results
from transformers import AutoTokenizer
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)





class ClassificationPipeline:
    """Main pipeline for running text classification experiments."""

    def __init__(
        self,
        dataset: BaseDataset,
        model: ScriptBaseModel,  # Use the renamed BaseModel
        parser: PredictionParser,
        converter: LabelConverter,
        evaluator: ClassificationEvaluator,
        batch_size: int = 8,  # Adjusted default for text
        num_workers: int = 4,
    ):
        self.dataset = dataset
        self.model = model
        self.parser = parser
        self.converter = converter
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.num_workers = num_workers

    # def custom_collate_fn(self, batch: List[Tuple[str, Dict, str, str, str]]):
    #     prompts, labels, item_ids, persona_ids, persona_pos = zip(*batch)
    #     return (
    #         list(prompts),
    #         list(labels),
    #         list(item_ids),
    #         list(persona_ids),
    #         list(persona_pos),
    #     )

    def run(self) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
        """Run the classification pipeline."""
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            # collate_fn=self.custom_collate_fn,  # Using a simple collate for text
        )

        results = []
        logger.info(
            f"Processing {len(self.dataset)} items in batches of {self.batch_size}..."
        )

        for (
            batch_prompts,
            batch_labels,
            batch_item_ids,
            batch_persona_ids,
            batch_persona_pos,
        ) in tqdm(dataloader):

            predictions = self.model.process_batch(batch_prompts)

            for idx, pred_json_str in enumerate(predictions):
                true_label_dict = self.converter.convert(batch_labels[idx])
                predicted_label_dict = self.parser.parse(pred_json_str)

                result = {
                    "item_id": batch_item_ids[idx],
                    "prompt": batch_prompts[idx],
                    "true_labels": true_label_dict,
                    "raw_prediction": pred_json_str,
                    "predicted_labels": predicted_label_dict,
                    "persona_id": batch_persona_ids[idx],
                    "persona_pos": batch_persona_pos[idx],
                }
                results.append(result)

                if len(results) % 300 == 0:
                    running_metrics = self.evaluator.calculate_metrics(results)
                    logger.info(f"\nProgress: {len(results)} items processed")
                    self._log_metrics(running_metrics)

        final_metrics = self.evaluator.calculate_metrics(results)
        return results, final_metrics

    def _log_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """Log metrics to console."""
        for aspect, aspect_metrics in metrics.items():
            logger.info(f"\n{aspect.upper()} metrics:")
            for metric_name, value in aspect_metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")


# ==================== Main ====================
def main():

    with open("config_text.yaml", "r") as f:
        config = yaml.safe_load(f)

    logger.info("Starting text classification pipeline...")
    logger.info(f"GPU State: {get_gpu_memory_info()}")

    # Process template strings in paths
    config["prompts_file"] = (
        config["prompts_file"]
        .replace("[DATASET]", "YODER")
        .replace("[MODEL_NAME]", config["model_id"].split("/")[-1])
    )

    config["output_path"] = (
        config["output_path"]
        .replace("[MODEL_NAME]", config["model_id"].split("/")[-1])
        .replace("[DATETIME]", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(config["output_path"]), exist_ok=True)

    logger.info(f"Output path: {config['output_path']}")
    logger.info(f"Using model: {config['model_id']}")

    model = VLLMModel(
        model_id=config["model_id"],
        additional_params={
            "seed": config["vllm_seed"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "max_model_len": config["max_model_len"],
            "max_num_seqs": config["max_num_seqs"],
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])

    # IMPORTANT: Ensure YoderIdentityDataset is modified to:
    # 1. Return raw prompt content (not pre-formatted with chat template).
    # 2. Return an item_id, e.g., str(item_idx) from its __getitem__.
    # Example __getitem__ return: (raw_prompt_content, label_dict, item_id, persona_id, persona_pos)
    dataset = YoderIdentityDataset(
        data_path=config["data_path"],
        tokenizer=tokenizer,
        prompts_file=config["prompts_file"],
        max_samples=config["max_samples"],
        seed=config["dataset_seed"],
    )

    parser = YoderPredictionParser()
    converter = YoderLabelConverter()
    evaluator = ClassificationEvaluator(aspects=["hate", "target"])

    pipeline = ClassificationPipeline(
        dataset=dataset,
        model=model,
        parser=parser,
        converter=converter,
        evaluator=evaluator,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    results, metrics = pipeline.run()

    logger.info("\nFinal Metrics:")
    pipeline._log_metrics(metrics)

    metadata = {
        "config": config,
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_class": model.__class__.__name__,
        "dataset_class": dataset.__class__.__name__,
    }

    save_results(results, metrics, config["output_path"], metadata)
    logger.info(f"Results saved to {config['output_path']}")

    logger.info("Pipeline completed successfully!")
    if hasattr(model, "cleanup"):
        model.cleanup()


if __name__ == "__main__":
    main()
