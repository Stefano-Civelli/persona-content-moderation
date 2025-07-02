import json
from typing import Any, Dict
from src.datasets.base import PredictionParser
from datasets.subdata_text_dataset import ContentClassification

class HateSpeechJsonParser(PredictionParser):
    """Parser for JSON-structured hate speech predictions."""
    
    def parse(self, prediction: str) -> Dict[str, Any]:
        """Parse JSON prediction into structured labels."""
        try:
            # Parse JSON
            classification = json.loads(prediction)
            
            # Validate against schema
            content_classification = ContentClassification(**classification)
            
            # Convert to dictionary with string values
            return {
                "is_hate_speech": content_classification.is_hate_speech.value,
                "target_category": content_classification.target_category.value,
                "specific_target": content_classification.specific_target.value
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            # Return default values on error
            return {
                "is_hate_speech": "no",
                "target_category": "none",
                "specific_target": "none"
            }
