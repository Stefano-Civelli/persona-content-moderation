import json
from typing import Dict, Any
import logging

from src.datasets.base import PredictionParser,

# Assuming YoderIdentityDataset and IdentityContentClassification are in yoder_text_dataset
from src.datasets.yoder_text_dataset import (
    IdentityContentClassification,
    isHateSpeech,
    IdentityTargetCategory,
    map_grouping,  # If needed by converter/parser
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class YoderPredictionParser(PredictionParser):
    """Parses VLLM's JSON output into IdentityContentClassification."""

    def parse(self, prediction: str) -> Dict[str, Any]:
        try:
            # Assuming prediction is a JSON string
            parsed_obj = IdentityContentClassification.model_validate_json(prediction)
            dumped_obj = parsed_obj.model_dump()
            return {
                "is_hate_speech": dumped_obj["is_hate_speech"] == isHateSpeech.true,
                "target_category": dumped_obj["target_category"].value,
            }
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON: {prediction}")
            # Return a default/error structure
            return {
                "is_hate_speech": False,
                "target_category": IdentityTargetCategory.none.value,
            }
        except Exception as e:
            logger.error(f"Error parsing prediction: {prediction}. Error: {e}")
            return {
                "is_hate_speech": False,
                "target_category": IdentityTargetCategory.none.value,
            }



