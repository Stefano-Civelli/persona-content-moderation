import vllm  # Ensure vLLM is imported before torch to avoid conflicts even if it's not used in this file
import logging
import argparse
from typing import Dict, Any, List, Tuple
import subprocess
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm


from src.datasets.multioff_dataset import MultiOffMemesDataset
from src.datasets.subdata_text_dataset import BinaryContentClassification
from src.models.base import BaseModel
from src.datasets.base import BaseDataset, PredictionParser
from src.datasets.facebook_hateful_memes_dataset_vllm import (
    FacebookHatefulMemesDataset,
    HatefulContentClassification,
)
from src.datasets.parser import HateSpeechJsonParser
from src.models.vision_model import Idefics3Model
from src.models.vllm_vision import Idefics3VLLMModel, QwenVL2_5VLLMModel
from utils.util import (
    ClassificationEvaluator,
    save_results,
    load_config,
    get_model_config,
)

import os

print(os.getcwd())

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ==================== Custom Collate ====================


def custom_collate_fn(batch):
    prompts, image, labels, item_ids, persona_ids, persona_pos = zip(*batch)
    return (
        list(prompts),
        list(image),
        list(labels),
        list(item_ids),
        list(persona_ids),
        list(persona_pos),
    )


# ==================== Pipeline ====================


class ClassificationPipeline:
    """Main pipeline for running classification experiments."""

    def __init__(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        parser: PredictionParser,
        evaluator: ClassificationEvaluator,
        output_path: str,
        batch_size: int = 2,
        num_workers: int = 4,
    ):
        self.dataset = dataset
        self.model = model
        self.parser = parser
        self.evaluator = evaluator
        self.output_path = output_path
        self.batch_size = batch_size
        self.num_workers = num_workers

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
            collate_fn=custom_collate_fn,
            prefetch_factor=1,  # Number of batches loaded in advance by each worker. ``2`` means there will be a total of 2 * num_workers batches prefetched across all workers. (default: ``2``)
        )

        all_results = []
        logger.info(
            f"Processing {len(self.dataset)} items in batches of {self.batch_size}..."
        )

        if start_batch > 0:
            logger.info(f"Starting from batch {start_batch}")

        first_batch = True
        for batch_idx, (
            unformatted_prompts,
            images,
            batch_labels,
            item_ids,
            persona_ids,
            persona_pos,
        ) in enumerate(tqdm(dataloader)):

            if batch_idx < start_batch:
                continue

            if first_batch:
                logger.info("=" * 70)
                logger.debug(f"unformatted prompts: {unformatted_prompts[:2]}")
                logger.debug(f"images: {images[:2]}")
                logger.debug(f"batch_labels: {batch_labels[:2]}")
                logger.debug(f"item_ids: {item_ids[:2]}")
                logger.debug(f"persona_ids: {persona_ids[:2]}")
                logger.debug(f"persona_pos: {persona_pos[:2]}")
                logger.info("=" * 70 + "\n")
                first_batch = False

            # The model's process_batch now handles formatting and vLLM interaction
            predictions = self.model.process_batch(unformatted_prompts, images)

            batch_results = []
            for idx, pred in enumerate(predictions):
                true_labels = self.dataset.convert_true_label(batch_labels[idx])
                predicted_labels = self.parser.parse(pred)

                result = {
                    "item_id": item_ids[idx],
                    "true_labels": true_labels,
                    "raw_prediction": pred,
                    "predicted_labels": predicted_labels,
                    "persona_id": persona_ids[idx],
                    "persona_pos": persona_pos[idx],
                }
                batch_results.append(result)

            # Save batch results
            start_item = batch_idx * self.batch_size
            end_item = start_item + len(unformatted_prompts) - 1
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
                logger.info(f"  {metric_name}: {value:.3f}")


# ==================== Main ====================
def main():
    # Find out which bunya node the code is running on
    result = subprocess.run(["hostname"], capture_output=True, text=True, check=True)
    print(result.stdout.strip())

    def none_or_int(value):
        if value == "None":
            return None
        return int(value)
    
    parser = argparse.ArgumentParser(description="Run content classification pipeline")
    parser.add_argument("--task_type", type=str, default="vision", help="Type of task")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model name/path", 
    )
    parser.add_argument(
        "--prompt_version",
        type=str,
        default="v1",  # v1, nopersona
    )
    parser.add_argument(
        "--extreme_personas_type",
        type=str,
        default="extreme_pos_corners_100",  # extreme_pos_left_right, extreme_pos_corners_100_centered
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="facebook",  # facebook, multioff
    )
    parser.add_argument(
        "--max_samples",
        type=none_or_int,
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
    args = parser.parse_args()

    config, prompt_templates = load_config()
    task_config = config[f"{args.task_type}_config"]
    model_config, task_config = get_model_config(task_config, args.model)
    if not model_config:
        logger.error(f"Model '{args.model}' not found in task '{args.task_type}'.")
        return

    prompt_template = prompt_templates[args.dataset_name][
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
        
    task_config["output_path"] = (
        task_config["output_path"]
        .replace("[MODEL_NAME]", MODEL.split("/")[-1])
        .replace("[DATETIME]", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
    )
    os.makedirs(os.path.dirname(task_config["output_path"]), exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"Extreme personas path: {task_config['extreme_pos_path']}")
    logger.info(f"Output path: {task_config['output_path']}")
    logger.info(f"Using model: {MODEL}")
    logger.info("=" * 70 + "\n")

    # ========== Create Objects ==========

    if "facebook" in args.dataset_name.lower():
        aspects = ["is_hate_speech", "target_group", "attack_method"]
        dataset = FacebookHatefulMemesDataset(
            data_path=task_config["data_path_facebook"],
            labels_relative_location=task_config["labels_relative_location_facebook"],
            max_samples=args.max_samples,
            extreme_pos_personas_path=extreme_pos_personas_path,
            prompt_template=prompt_template,
        )
        json_schema_class = HatefulContentClassification
    elif "multioff" in args.dataset_name.lower():
        aspects = ["is_hate_speech"]
        dataset = MultiOffMemesDataset(
            data_path=task_config["data_path_multioff"],
            labels_relative_location=task_config["labels_relative_location_multioff"],
            max_samples=args.max_samples,
            extreme_pos_personas_path=extreme_pos_personas_path,
            prompt_template=prompt_template,
        )
        json_schema_class = BinaryContentClassification
    else:
        logger.error(f"Unknown dataset name: {args.dataset_name}")
        return

    logger.info(f"Dataset size: {len(dataset)}")


    if "idefics3" in MODEL.lower():
        model = Idefics3VLLMModel(
            model_id=MODEL,
            seed=model_config["vllm_seed"],
            temperature=model_config["temperature"],
            enforce_eager=model_config["enforce_eager"],
            resolution_factor=model_config["resolution_factor"],
            max_tokens=model_config["max_tokens"],
            max_model_len=model_config["max_model_len"],
            max_num_seqs=model_config["max_num_seqs"],
            json_schema_class=json_schema_class,
        )
    elif "qwen" in MODEL.lower():
        model = QwenVL2_5VLLMModel(
            model_id=MODEL,
            seed=model_config["vllm_seed"],
            temperature=model_config["temperature"],
            enforce_eager=model_config["enforce_eager"],
            max_tokens=model_config["max_tokens"],
            max_model_len=model_config["max_model_len"],
            max_num_seqs=model_config["max_num_seqs"],
            json_schema_class=json_schema_class,
        )
    else:
        raise ValueError(f"Model ID {MODEL} not supported by this script.")

    
    parser = HateSpeechJsonParser(json_schema_class=json_schema_class)
    evaluator = ClassificationEvaluator(aspects=aspects)

    # Create and run pipeline
    pipeline = ClassificationPipeline(
        dataset=dataset,
        model=model,
        parser=parser,
        evaluator=evaluator,
        output_path=task_config["output_path"],
        batch_size=task_config["batch_size"],
        num_workers=task_config["num_workers"],
    )
    # ========== Create Objects ==========

    results, metrics = pipeline.run()

    logger.info("\nFinal Metrics:")
    pipeline._log_metrics(metrics)

    # Save results
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
