import logging
import vllm  # Ensure vLLM is imported before torch to avoid conflicts even if it's not used in this file

# os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# import multiprocessing
from typing import Dict, Any, List, Tuple
import yaml

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src.datasets.subdata_text_dataset import SubdataTextDataset
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
from utils.util import ClassificationEvaluator, save_results
from transformers import AutoTokenizer


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

    def custom_collate_fn(self, batch: List[Tuple[str, Dict, str, str, str]]):
        prompts, labels, item_ids, persona_ids, persona_pos = zip(*batch)
        return (
            list(prompts),
            list(labels),
            list(item_ids),
            list(persona_ids),
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
            collate_fn=self.custom_collate_fn,  # Using a simple collate for text
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
                logger.debug(f"Batch prompts: {batch_prompts[:1]}")
                logger.debug(f"Batch labels: {batch_labels[:1]}")
                logger.debug(f"Batch item IDs: {batch_item_ids[:1]}")
                logger.debug(f"Batch persona IDs: {batch_persona_ids[:1]}")
                logger.debug(f"Batch persona positions: {batch_persona_pos[:1]}")
                first_batch = False

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
    # multiprocessing.set_start_method("spawn", force=True)

    with open("config_text.yaml", "r") as f:
        config = yaml.safe_load(f)

    MODEL = config["models"][config["model_id"]]
    print(type(MODEL))

    # Process template strings in paths
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
    logger.info(f"Using model: {MODEL}")

    model = VLLMModel(
        model_id=MODEL,
        seed=config["vllm_seed"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_model_len=config["max_model_len"],
        max_num_seqs=config["max_num_seqs"],
        enforce_eager=config["enforce_eager"],
    )
    print("=" * 40)
    print("========= Model setup complete =========")
    print("=" * 40)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    dataset = YoderIdentityDataset(
        data_path=config["data_path"],
        tokenizer=tokenizer,
        prompts_file=config["prompts_file"],
        max_samples=config["max_samples"],
        seed=config["dataset_seed"],
        fold=config["fold"],
        target_group_size=config["target_group_size"],
    )

    # dataset = SubdataTextDataset(
    #     data_path=config["data_path"],
    #     tokenizer=tokenizer,
    #     prompts_file=config["prompts_file"],
    #     max_samples=config["max_samples"],
    #     seed=config["dataset_seed"],
    #     split="gender",
    # )

    # print some debugging information
    logger.info(f"Dataset size: {len(dataset)}")

    parser = YoderPredictionParser()
    converter = YoderLabelConverter()
    evaluator = ClassificationEvaluator(aspects=["is_hate_speech", "target_category"])

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
