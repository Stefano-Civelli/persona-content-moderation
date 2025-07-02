from typing import Any, Dict, Optional, Tuple
import json
from pathlib import Path
import pandas as pd
from PIL import Image
from src.datasets.base import BaseDataset


class FacebookHatefulMemesDataset(BaseDataset):
    """Dataset implementation for Facebook Hateful Memes."""

    def __init__(
        self,
        data_path: str,
        labels_relative_location: str,
        processor,
        prompts_file: str,
        max_samples: Optional[int] = None,
        extreme_pos_personas_path: Optional[str] = None,
        prompt_template: Optional[str] = None,
        seed: int = 42,
        **additional_params: Any,
    ):
        self.processor = processor
        self.labels_relative_location = labels_relative_location
        self.prompts = {}
        self.persona_ids = []

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
        labels_path = Path(self.data_path) / self.labels_relative_location

        with open(labels_path, "r") as f:
            data = [json.loads(line) for line in f]

        # filter away data items where gold_pc or gold_attack contain more than one label
        total_before = len(data)
        data = [
            item
            for item in data
            if len(item["gold_pc"]) == 1 and len(item["gold_attack"]) == 1
        ]
        ignored_items = total_before - len(data)
        print(
            f"===Ignored {ignored_items} items due to multiple labels in gold_pc or gold_attack==="
        )

        if self.max_samples:
            import random

            random.seed(self.seed)
            data = random.sample(data, min(self.max_samples, len(data)))

        for item in data:
            img_path = Path(self.data_path) / item["img"]
            if img_path.exists():
                self.items.append(
                    {
                        "image_path": str(img_path),
                        "labels": {
                            "hate": item["gold_hate"],
                            "pc": item["gold_pc"],
                            "attack": item["gold_attack"],
                        },
                        "item_id": item["img"],
                    }
                )

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, str, Dict]:
        num_prompts = len(self.prompts)
        item_idx = idx // num_prompts
        prompt_idx = idx % num_prompts

        item = self.items[item_idx]
        persona_id = self.persona_ids[prompt_idx]
        prompt, persona_pos = self.prompts[persona_id]

        # Load and process image
        image = Image.open(item["image_path"]).convert("RGB")

        # Prepare message
        message = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }

        # Process with model processor
        prompt_text = self.processor.apply_chat_template(
            [message], add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt_text, images=[image], padding=True, return_tensors="pt"
        )

        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return inputs, item["labels"], item["item_id"], persona_id, persona_pos

    def __len__(self) -> int:
        """Return total number of items including prompt variations."""
        if self.prompts:
            return len(self.items) * len(self.prompts)
        return len(self.items)
