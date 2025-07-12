import json
from typing import Any, Dict
from src.datasets.base import PredictionParser
import logging

logger = logging.getLogger(__name__)


class HateSpeechJsonParser(PredictionParser):
    """Parser for JSON-structured hate speech predictions."""

    def __init__(self, json_schema_class):
        self.json_schema_class = json_schema_class

    def parse(self, prediction: str, none_if_false: bool = False) -> Dict[str, Any]:
        """Parse JSON prediction into structured labels."""
        try:
            parsed_obj = self.json_schema_class.model_validate_json(prediction)
            dumped_obj = parsed_obj.model_dump()

            for i, (key, value) in enumerate(dumped_obj.items()):
                if i == 0:
                    dumped_obj[key] = value.value.lower() == "true"
                    if (none_if_false) and (not dumped_obj[key]):
                        # if first value is false
                        for remaining_key in list(dumped_obj.keys())[1:]:
                            dumped_obj[remaining_key] = "none"
                        break
                elif value is None:
                    dumped_obj[key] = "none"
                elif hasattr(value, "value"):
                    if value.value is None:
                        dumped_obj[key] = "none"
                    else:
                        dumped_obj[key] = value.value

            return dumped_obj

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                f"Failed to parse or validate prediction. Error: {e}\n"
                # f"Prediction was: ---{prediction}---"
            )
            # In failure cases, return false for first field and NONE for others
            schema_fields = self.json_schema_class.model_json_schema()[
                "properties"
            ].keys()
            result = {}
            for i, field in enumerate(schema_fields):
                result[field] = False if i == 0 else "none"
            return result
