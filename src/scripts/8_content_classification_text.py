import logging
import vllm  # Ensure vLLM is imported before torch to avoid conflicts even if it's not used in this file
from typing import Dict, Any, List, Tuple

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
from src.datasets.parser import HateSpeechJsonParser
from src.datasets.base import BaseDataset, PredictionParser
from src.datasets.subdata_text_dataset import SubdataTextDataset
from src.models.base import (
    BaseModel as ScriptBaseModel,
)  # Renamed to avoid conflict with pydantic.BaseModel


from src.models.vllm_model import VLLMModel

# Assuming YoderIdentityDataset and IdentityContentClassification are in yoder_text_dataset
from src.datasets.yoder_text_dataset import (
    IdentityContentClassification,
    YoderIdentityDataset,
)
from utils.util import (
    ClassificationEvaluator,
    save_results,
    load_config,
    get_model_config,
)
from transformers import AutoTokenizer


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ==================== Custom Collate ====================


def custom_collate_fn(batch: List[Tuple[str, Dict, str, str, str]]):
    prompts, labels, item_ids, persona_ids, persona_pos = zip(*batch)
    return (
        list(prompts),
        list(labels),
        list(item_ids),
        list(persona_ids),
        list(persona_pos),
    )


# ==================== Pipeline ====================


class ClassificationPipeline:
    """Main pipeline for running text classification experiments."""

    def __init__(
        self,
        dataset: BaseDataset,
        model: ScriptBaseModel,  # Use the renamed BaseModel
        parser: PredictionParser,
        evaluator: ClassificationEvaluator,
        batch_size: int = 8,  # Adjusted default for text
        num_workers: int = 4,
    ):
        self.dataset = dataset
        self.model = model
        self.parser = parser
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.num_workers = num_workers

    def run(self) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
        """Run the classification pipeline."""
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=custom_collate_fn,  # Using a simple collate for text
        )

        results = []
        logger.info(
            f"Processing {len(self.dataset)} items in batches of {self.batch_size}..."
        )

        first_batch = True
        for (
            batch_prompts,
            batch_labels,
            batch_item_ids,
            batch_persona_ids,
            batch_persona_pos,
        ) in tqdm(dataloader):

            if first_batch:
                logger.info("=" * 70)
                logger.debug(f"Batch prompts: {batch_prompts[:2]}")
                logger.debug(f"Batch labels: {batch_labels[:2]}")
                logger.debug(f"Batch item IDs: {batch_item_ids[:2]}")
                logger.debug(f"Batch persona IDs: {batch_persona_ids[:2]}")
                logger.debug(f"Batch persona positions: {batch_persona_pos[:2]}")
                logger.info("=" * 70 + "\n")
                first_batch = False

            predictions = self.model.process_batch(batch_prompts)

            for idx, pred_json_str in enumerate(predictions):
                true_label_dict = self.dataset.convert_true_label(batch_labels[idx])
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
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run content classification pipeline")
    parser.add_argument("--task_type", type=str, default="text", help="Type of task")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name/path",
    )
    parser.add_argument(
        "--prompt_version",
        type=str,
        default="v2",
    )
    parser.add_argument(
        "--extreme_personas_type",
        type=str,
        default="extreme_pos_corners_100",  # extreme_pos_left_right
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--run_description",
        type=str,
        default="",
        help="A short description for the run, to be saved in the metadata.",
    )
    args = parser.parse_args()

    config, prompt_templates = load_config()
    task_config = config[f"{args.task_type}_config"]
    model_config, task_config = get_model_config(task_config, args.model)
    if not model_config:
        logger.error(f"Model '{args.model}' not found in task '{args.task_type}'.")
        return

    prompt_template = prompt_templates[task_config["dataset_name"]][
        args.prompt_version
    ]["template"]

    MODEL = args.model

    task_config["extreme_pos_path"] = (
        task_config["extreme_pos_path"]
        .replace("[MODEL_NAME]", MODEL.split("/")[-1])
        .replace("[TYPE]", args.extreme_personas_type)
    )
    task_config["output_path"] = (
        task_config["output_path"]
        .replace("[MODEL_NAME]", MODEL.split("/")[-1])
        .replace("[DATETIME]", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
    )
    os.makedirs(os.path.dirname(task_config["output_path"]), exist_ok=True)

    if "yoder" in task_config["dataset_name"].lower():
        aspects = ["is_hate_speech", "target_category"]
    elif "subdata" in task_config["dataset_name"].lower():
        aspects = ["is_hate_speech"]
    else:
        logger.error(f"Unknown dataset name: {task_config['dataset_name']}")
        return

    logger.info("=" * 70)
    logger.info(f"Extreme personas path: {task_config['extreme_pos_path']}")
    logger.info(f"Output path: {task_config['output_path']}")
    logger.info(f"Using model: {MODEL}")
    logger.info(f"Aspects being evaluated: {aspects}")
    logger.info("=" * 70 + "\n")

    # ========== Create Objects ==========

    model = VLLMModel(
        model_id=MODEL,
        seed=model_config["vllm_seed"],
        temperature=model_config["temperature"],
        max_model_len=model_config["max_model_len"],
        max_num_seqs=model_config["max_num_seqs"],
        max_tokens=model_config["max_tokens"],
        enforce_eager=model_config["enforce_eager"],
        tensor_parallel_size=model_config.get("tensor_parallel_size", 1),
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    dataset = YoderIdentityDataset(
        data_path=task_config["data_path"],
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        seed=task_config["dataset_seed"],
        fold=task_config["fold"],
        target_group_size=task_config["target_group_size"],
        extreme_pos_personas_path=task_config["extreme_pos_path"],
        prompt_template=prompt_template,
    )

    # dataset = SubdataTextDataset(
    #     data_path=config["data_path"],
    #     tokenizer=tokenizer,
    #     max_samples=args.max_samples,
    #     seed=config["dataset_seed"],
    #     split="gender",
    # )

    # print some debugging information
    logger.info(f"Dataset size: {len(dataset)}")

    parser = HateSpeechJsonParser(json_schema_class=IdentityContentClassification)

    evaluator = ClassificationEvaluator(aspects=aspects)

    pipeline = ClassificationPipeline(
        dataset=dataset,
        model=model,
        parser=parser,
        evaluator=evaluator,
        batch_size=task_config["batch_size"],
        num_workers=task_config["num_workers"],
    )

    results, metrics = pipeline.run()

    logger.info("\nFinal Metrics:")
    pipeline._log_metrics(metrics)

    metadata = {
        "task_config": task_config,
        "model_config": model_config,
        "run_description": args.run_description,
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_class": model.__class__.__name__,
        "dataset_class": dataset.__class__.__name__,
    }

    save_results(results, metrics, task_config["output_path"], metadata)
    logger.info(f"Results saved to {task_config['output_path']}")

    logger.info("Pipeline completed successfully!")
    if hasattr(model, "cleanup"):
        model.cleanup()


if __name__ == "__main__":
    main()
