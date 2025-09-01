import json
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple
import random
from pydantic import BaseModel
from enum import Enum

from src.datasets.base import BaseDataset


class isHateSpeech(str, Enum):
    true = "true"
    false = "false"


class TargetGroup(str, Enum):
    disability = "disability"
    race = "race"
    religion = "religion"
    nationality = "nationality"
    sex = "sex"
    none = "none"


class AttackMethod(str, Enum):
    contempt = "contempt"
    mocking = "mocking"
    inferiority = "inferiority"
    slurs = "slurs"
    exclusion = "exclusion"
    dehumanizing = "dehumanizing"
    inciting_violence = "inciting_violence"
    none = "none"


# class HatefulContentClassification(BaseModel):
#     is_hate_speech: isHateSpeech
#     target_group: TargetGroup
#     attack_method: AttackMethod


# NOTE This seems to be betterat actually getting 
class HatefulContentClassification(BaseModel):
    is_hate_speech: isHateSpeech
    target_group: Optional[TargetGroup] = None
    attack_method: Optional[AttackMethod] = None


class FacebookHatefulMemesDataset(BaseDataset):
    def __init__(
        self,
        data_path: str,
        labels_relative_location: List[str],
        prompts_file: Optional[str] = None,
        max_samples: Optional[int] = None,
        extreme_pos_personas_path: Optional[str] = None,
        prompt_template: Optional[str] = None,
        seed: int = 42,
        **additional_params: Any,
    ):
        # The processor is no longer needed here, as vLLM handles it.
        self.labels_relative_location = labels_relative_location
        self.persona_ids = []
        self.items = []

        super().__init__(
            data_path,
            prompts_file,
            max_samples,
            extreme_pos_personas_path,
            prompt_template,
            seed,
            **additional_params,
        )

    def load_dataset(self) -> None:
        """Load Facebook Hateful Memes dataset."""
        labels_paths = [Path(self.data_path) / location for location in self.labels_relative_location]

        data = []
        for labels_path in labels_paths:
            with open(labels_path, "r") as f:
                data.extend([json.loads(line) for line in f])

        # total_before = len(data)
        # data = [
        #     item
        #     for item in data
        #     if len(item["gold_pc"]) == 1 and len(item["gold_attack"]) == 1
        # ]
        # ignored_items = total_before - len(data)
        # print(
        #     f"===Ignored {ignored_items} items due to multiple labels in gold_pc or gold_attack==="
        # )


        if self.max_samples:
            random.seed(self.seed)
            data = random.sample(data, min(self.max_samples, len(data)))

        seen_images = set()
        for item in data:
            img_path = Path(self.data_path) / item["img"]
            if img_path.exists() and img_path not in seen_images:
                self.items.append({
                    "image_path": str(img_path),
                    "labels": {
                        "hate": item["gold_hate"],
                        "pc": item["gold_pc"],
                        "attack": item["gold_attack"],
                    },
                    "item_id": item["img"],
                })
                seen_images.add(img_path)
            else:
                print(f"Image {img_path} does not exist or duplicate. Skipping item.")

        
        print(f"loaded a total of {len(self.items)} images")

    def __getitem__(self, idx: int) -> Tuple[str, Image.Image, Dict, str, str, str]:

        # num_prompts = len(self.prompts)
        # prompt_idx = idx % num_prompts
        # item_idx = idx // num_prompts
        # TODO understand if this works better or worse for images
        num_images = len(self.items)
        prompt_idx = idx // num_images
        item_idx = idx % num_images

        item = self.items[item_idx]
        persona_id = self.persona_ids[prompt_idx]
        unformatted_prompt, persona_pos = self.prompts[persona_id]

        # Load image as a PIL object
        image = Image.open(item["image_path"]).convert("RGB")

        # Return the raw components. The model class will format them.
        return (
            unformatted_prompt,
            image,
            item["labels"],
            item["item_id"],
            persona_id,
            persona_pos,
        )

    def __len__(self) -> int:
        """Return total number of items including prompt variations."""
        if self.prompts:
            return len(self.items) * len(self.prompts)
        return len(self.items)
    
    # TODO it should be possible to eliminate this if I change the names above
    def convert_true_label(self, raw_label):
        hate_label = (
            raw_label["hate"][0]
            if raw_label.get("hate")
            else "not_hateful"
        )
        pc_label = (
            raw_label["pc"][0]
            if raw_label.get("pc")
            else "pc_empty"
        )
        attack_label = (
            raw_label["attack"][0]
            if raw_label.get("attack")
            else "attack_empty"
        )

        return {
            "is_hate_speech": hate_label != "not_hateful",
            "target_group": (
                pc_label.replace("pc_", "") if pc_label != "pc_empty" else "none"
            ),
            "attack_method": (
                attack_label.replace("attack_", "")
                if attack_label != "attack_empty"
                else "none"
            ),
        }