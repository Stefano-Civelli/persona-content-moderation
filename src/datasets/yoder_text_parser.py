import json
from typing import Dict, Any
import logging

from src.datasets.base import (
    PredictionParser,
    LabelConverter,
)

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
            return parsed_obj.model_dump()
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON: {prediction}")
            # Return a default/error structure
            return {
                "is_hate_speech": isHateSpeech.no,
                "target_category": IdentityTargetCategory.none,
            }
        except Exception as e:
            logger.error(f"Error parsing prediction: {prediction}. Error: {e}")
            return {
                "is_hate_speech": isHateSpeech.no,
                "target_category": IdentityTargetCategory.none,
            }


class YoderLabelConverter(LabelConverter):
    """Converts Yoder dataset labels to a consistent dictionary format for evaluation."""

    def convert(self, label: Dict[str, Any]) -> Dict[str, Any]:
        # Dataset label format: {"hate": "yes"/"no", "target": "category_string"}
        # Output format for evaluation: {"is_hate_speech": "yes/no", "target_category": "category_string"}
        return {
            "is_hate_speech": label.get("hate", isHateSpeech.no.value),
            "target_category": label.get("target", IdentityTargetCategory.none.value),
        }