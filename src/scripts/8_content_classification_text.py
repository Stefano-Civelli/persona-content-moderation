import logging
import vllm  # Ensure vLLM is imported before torch to avoid conflicts even if it's not used in this file
from typing import Dict, Any, List, Tuple
import json
import glob


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
import subprocess

from src.models.vllm_model import VLLMModel

# Assuming YoderIdentityDataset and IdentityContentClassification are in yoder_text_dataset
from src.datasets.yoder_text_dataset import (
    IdentityContentClassification,
    YoderIdentityDataset,
)
from src.datasets.subdata_text_dataset import BinaryContentClassification
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
        output_path: str,
        batch_size: int = 8,  # Adjusted default for text
        num_workers: int = 4,
    ):
        self.dataset = dataset
        self.model = model
        self.parser = parser
        self.evaluator = evaluator
        self.output_path = output_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def load_previous_batch_results(self, start_batch: int) -> List[Dict[str, Any]]:
        """Load results from previous batches when resuming from a checkpoint."""
        all_previous_results = []
        batch_dir = os.path.join(self.output_path, "batches")
        
        if not os.path.exists(batch_dir):
            logger.warning(f"Batch directory {batch_dir} does not exist. Starting with empty results.")
            return all_previous_results
        
        logger.info(f"Loading previous batch results from {batch_dir}")
        
        # Get all batch files and sort them by start item number
        all_batch_files = glob.glob(os.path.join(batch_dir, "results_*.json"))
        batch_files_info = []
        
        for file_path in all_batch_files:
            filename = os.path.basename(file_path)
            try:
                # Extract start and end items from filename like "results_0-49999.json"
                range_part = filename.replace("results_", "").replace(".json", "")
                start_item, end_item = map(int, range_part.split('-'))
                batch_files_info.append((start_item, end_item, file_path))
            except (IndexError, ValueError) as e:
                logger.warning(f"Could not parse filename {filename}: {e}")
                continue
        
        # Sort by start item
        batch_files_info.sort(key=lambda x: x[0])
        
        # Only load batches that come before start_batch
        start_item_threshold = start_batch * self.batch_size
        
        for start_item, end_item, file_path in batch_files_info:
            # Only load if this batch is completely before our start_batch
            if start_item < start_item_threshold:
                logger.info(f"Loading batch file {os.path.basename(file_path)} (items {start_item}-{end_item})")
                try:
                    with open(file_path, 'r') as f:
                        batch_data = json.load(f)
                        # Handle different possible structures
                        if isinstance(batch_data, list):
                            all_previous_results.extend(batch_data)
                        elif isinstance(batch_data, dict):
                            if 'results' in batch_data:
                                all_previous_results.extend(batch_data['results'])
                            elif 'data' in batch_data:
                                all_previous_results.extend(batch_data['data'])
                            else:
                                logger.warning(f"Unknown structure in {file_path}, trying to use as list")
                                all_previous_results.extend([batch_data])
                except Exception as e:
                    logger.error(f"Error loading batch file {file_path}: {e}")
                    continue
            else:
                logger.info(f"Skipping batch file {os.path.basename(file_path)} (starts at item {start_item}, beyond threshold {start_item_threshold})")
        
        logger.info(f"Loaded {len(all_previous_results)} results from previous batches")
        return all_previous_results

    def run(
        self, start_batch: int = 0
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
        """Run the classification pipeline."""
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=custom_collate_fn,  # Using a simple collate for text
        )

        # Load previous results if starting from a later batch
        all_results = []
        if start_batch > 0:
            all_results = self.load_previous_batch_results(start_batch)
            logger.info(f"Resuming from batch {start_batch} with {len(all_results)} previous results")
        
        logger.info(
            f"Processing {len(self.dataset)} items in batches of {self.batch_size}..."
        )
        if start_batch > 0:
            logger.info(f"Starting from batch {start_batch}")

        first_batch = True
        for batch_idx, (
            batch_prompts,
            batch_labels,
            batch_item_ids,
            batch_persona_ids,
            batch_persona_pos,
        ) in enumerate(tqdm(dataloader)):

            if batch_idx < start_batch:
                continue

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

            batch_results = []
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
                batch_results.append(result)

            # Save batch results
            start_item = batch_idx * self.batch_size
            end_item = start_item + len(batch_prompts) - 1
            batch_filename = os.path.join(
                self.output_path, "batches", f"results_{start_item}-{end_item}.json"
            )
            save_results(batch_results, {}, batch_filename, {})
            logger.info(f"Saved batch {batch_idx} results to {batch_filename}")

            all_results.extend(batch_results)

            running_metrics = self.evaluator.calculate_metrics(all_results)
            logger.info(f"\nProgress: {len(all_results)} items processed")
            self._log_metrics(running_metrics)

        final_metrics = self.evaluator.calculate_metrics(all_results)
        return all_results, final_metrics

    def _log_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """Log metrics to console."""
        for aspect, aspect_metrics in metrics.items():
            logger.info(f"\n{aspect.upper()} metrics:")
            for metric_name, value in aspect_metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")


# ==================== Main ====================
def main():
    # Find out which bunya node the code is running on
    result = subprocess.run(["hostname"], capture_output=True, text=True, check=True)
    print(result.stdout.strip())

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
        default="nopersona", # v1, nopersona
    )
    parser.add_argument(
        "--extreme_personas_type",
        type=str,
        default=None,  # None, extreme_pos_corners_100_centered
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="subdata",  # subdata, yoder, cad
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--start_batch",
        type=int,
        default=0,
        help="The batch number to start processing from.",
    )
    parser.add_argument(
        "--run_description",
        type=str,
        default="",
        help="A short description for the run, to be saved in the metadata.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Timestamp folder name to resume from (required when start_batch > 0)",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.start_batch > 0 and args.timestamp is None:
        logger.error("--timestamp is required when --start_batch > 0")
        return

    config, prompt_templates = load_config()
    task_config = config[f"{args.task_type}_config"]
    model_config, task_config = get_model_config(task_config, args.model)
    if not model_config:
        logger.error(f"Model '{args.model}' not found in task '{args.task_type}'.")
        return
    

    template_name = args.dataset_name if args.dataset_name != 'cad' else "subdata"
    prompt_template = prompt_templates[template_name][
        args.prompt_version
    ]["template"]

    MODEL = args.model

    extreme_pos_personas_path = None
    if args.extreme_personas_type is not None:
        extreme_pos_personas_path = (
            task_config["extreme_pos_path"]
            .replace("[MODEL_NAME]", MODEL.split("/")[-1])
            .replace("[TYPE]", args.extreme_personas_type)
        )
        task_config["extreme_pos_path"] = extreme_pos_personas_path
    
    # Handle output path - use provided timestamp or create new one
    if args.timestamp:
        # Use the provided timestamp for resuming
        task_config["output_path"] = (
            task_config["output_path"]
            .replace("[MODEL_NAME]", MODEL.split("/")[-1])
            .replace("[DATETIME]", args.timestamp)
        )
        logger.info(f"Resuming from timestamp: {args.timestamp}")
    else:
        # Create new timestamp for new run
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        task_config["output_path"] = (
            task_config["output_path"]
            .replace("[MODEL_NAME]", MODEL.split("/")[-1])
            .replace("[DATETIME]", timestamp)
        )
        logger.info(f"Starting new run with timestamp: {timestamp}")
    
    # Create output directories
    os.makedirs(task_config["output_path"], exist_ok=True)
    os.makedirs(os.path.join(task_config["output_path"], "batches"), exist_ok=True)

    logger.info("=" * 70)
    if extreme_pos_personas_path:
        logger.info(f"Extreme personas path: {extreme_pos_personas_path}")
    else:
        logger.info("Running in baseline mode (no personas)")
    logger.info(f"Output path: {task_config['output_path']}")
    logger.info(f"Using model: {MODEL}")
    logger.info("=" * 70 + "\n")

    # ========== Create Objects ==========

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    if "yoder" in args.dataset_name.lower():
        aspects = ["is_hate_speech", "target_category"]

        dataset = YoderIdentityDataset(
            data_path=task_config["data_path_yoder"],
            tokenizer=tokenizer,
            max_samples=args.max_samples,
            seed=task_config["dataset_seed"],
            fold=task_config["fold"],
            target_group_size=task_config["target_group_size"],
            extreme_pos_personas_path=extreme_pos_personas_path,
            prompt_template=prompt_template,
        )
        json_schema_class = IdentityContentClassification
    elif "subdata" in args.dataset_name.lower() or "cad" in args.dataset_name.lower():
        aspects = ["is_hate_speech"]
        if args.dataset_name == "subdata":
            data_path = "data_path_subdata"
        elif args.dataset_name == "cad":
            data_path = "data_path_cad"
        else:
            raise ValueError(f"Unsupported dataset_name: '{args.dataset_name}'. Expected 'subdata' or 'cad'.")

        dataset = SubdataTextDataset(
            data_path=task_config[data_path],
            tokenizer=tokenizer,
            max_samples=args.max_samples,
            extreme_pos_personas_path=extreme_pos_personas_path,
            prompt_template=prompt_template,
            seed=task_config["dataset_seed"],
            split="political",
        )
        json_schema_class = BinaryContentClassification
    else:
        logger.error(f"Unknown dataset name: {args.dataset_name}")
        return

    logger.info(f"Dataset size: {len(dataset)}")

    model = VLLMModel(
        model_id=MODEL,
        seed=model_config["vllm_seed"],
        temperature=model_config["temperature"],
        max_model_len=model_config["max_model_len"],
        max_num_seqs=model_config["max_num_seqs"],
        max_tokens=model_config["max_tokens"],
        enforce_eager=model_config["enforce_eager"],
        tensor_parallel_size=model_config.get("tensor_parallel_size", 1),
        json_schema_class=json_schema_class,
    )

    parser = HateSpeechJsonParser(json_schema_class=json_schema_class)
    evaluator = ClassificationEvaluator(aspects=aspects)

    pipeline = ClassificationPipeline(
        dataset=dataset,
        model=model,
        parser=parser,
        evaluator=evaluator,
        output_path=task_config["output_path"],
        batch_size=task_config["batch_size"],
        num_workers=task_config["num_workers"],
    )

    results, metrics = pipeline.run(start_batch=args.start_batch)

    logger.info("\nFinal Metrics:")
    pipeline._log_metrics(metrics)

    metadata = {
        "task_config": task_config,
        "model_config": model_config,
        "dataset_name": args.dataset_name,
        "prompt_version": args.prompt_version,
        "extreme_personas_type": args.extreme_personas_type,
        "max_samples": args.max_samples,
        "run_description": args.run_description,
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_class": model.__class__.__name__,
        "dataset_class": dataset.__class__.__name__,
        "start_batch": args.start_batch,
        "resumed_from_timestamp": args.timestamp,
        "baseline_mode": extreme_pos_personas_path is None,
    }

    final_results_path = os.path.join(task_config["output_path"], "final_results.json")
    save_results(results, metrics, final_results_path, metadata)
    logger.info(f"Results saved to {final_results_path}")

    logger.info("Pipeline completed successfully!")
    if hasattr(model, "cleanup"):
        model.cleanup()


if __name__ == "__main__":
    main()