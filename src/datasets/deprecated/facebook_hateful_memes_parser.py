from typing import Any, Dict
from src.datasets.base import PredictionParser


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
        labels = {"is_hate_speech": False, "target_group": "none", "attack_method": "none"}

        pred = prediction.strip().rstrip(".")

        if pred == "NOT_HARMFUL":
            return labels

        if pred.startswith("HARMFUL"):
            labels["is_hate_speech"] = True

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



