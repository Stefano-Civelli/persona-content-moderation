from typing import Any, Dict
from src.datasets.base import PredictionParser, LabelConverter

class HatefulMemesPredictionParser(PredictionParser):
    """Parser for hateful memes predictions."""

    VALID_TARGET_GROUPS = {"disability", "race", "religion", "nationality", "sex"}
    VALID_ATTACK_METHODS = {
        "contempt",
        "mocking",
        "inferiority",
        "slurs",
        "exclusion",
        "dehumanizing",
        "inciting_violence",
    }

    def parse(self, prediction: str) -> Dict[str, Any]:
        """Parse Idefics3 prediction for hateful memes."""
        labels = {"harmful": False, "target_group": "none", "attack_method": "none"}

        pred = prediction.strip().rstrip(".")

        if pred == "NOT_HARMFUL":
            return labels

        if pred.startswith("HARMFUL"):
            labels["harmful"] = True

            for line in pred.split("\n"):
                line = line.strip()

                if line.startswith("TG:"):
                    tg = line.replace("TG:", "").strip().lower()
                    for group in self.VALID_TARGET_GROUPS:
                        if group in tg:
                            labels["target_group"] = group
                            break

                elif line.startswith("AM:"):
                    am = line.replace("AM:", "").strip().lower()
                    for method in self.VALID_ATTACK_METHODS:
                        if method in am:
                            labels["attack_method"] = method
                            break

        return labels


class HatefulMemesLabelConverter(LabelConverter):
    """Label converter for hateful memes dataset."""

    def convert(self, raw_labels: Dict) -> Dict[str, Any]:
        """Convert raw labels to standardized format."""
        hate_labels = raw_labels["hate"][0] if raw_labels["hate"] else "not_hateful"
        pc_labels = raw_labels["pc"][0] if raw_labels["pc"] else "pc_empty"
        attack_labels = (
            raw_labels["attack"][0] if raw_labels["attack"] else "attack_empty"
        )

        return {
            "harmful": hate_labels != "not_hateful",
            "target_group": (
                pc_labels.replace("pc_", "") if pc_labels != "pc_empty" else "none"
            ),
            "attack_method": (
                attack_labels.replace("attack_", "")
                if attack_labels != "attack_empty"
                else "none"
            ),
        }