from typing import Any, Dict, Optional, Tuple
import json
from pathlib import Path
import pandas as pd
from PIL import Image
from src.datasets.base import BaseDataset


class FacebookHatefulMemesDataset(BaseDataset):
    """Dataset implementation for Facebook Hateful Memes."""

    def __init__(
        self, data_path: str, processor, prompts_file: str, max_samples: Optional[int] = None,seed: int = 42, **additional_params: Any
    ):
        self.processor = processor
        self.prompts_file = prompts_file
        self.prompts = {}
        self.persona_ids = []
        self._load_prompts()

        super().__init__(data_path, max_samples, seed, **additional_params)

    def _load_prompts(self) -> None:
        """Load prompts from file."""
        prompts_df = pd.read_parquet(self.prompts_file)
        for _, row in prompts_df.iterrows():
            self.prompts[row["persona_id"]] = (row["prompt"], row["persona_pos"])
        self.persona_ids = list(self.prompts.keys())

    def load_dataset(self) -> None:
        """Load Facebook Hateful Memes dataset."""
        train_path = Path(self.data_path) / "fine_grained_labels" / "train.jsonl"

        with open(train_path, "r") as f:
            data = [json.loads(line) for line in f]

        # Filter single-label items
        data = [
            item
            for item in data
            if len(item["gold_pc"]) == 1 and len(item["gold_attack"]) == 1
        ]

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

        metadata = {"persona_id": persona_id, "persona_pos": persona_pos}

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

        return inputs, item["labels"], item["item_id"], metadata

    def __len__(self) -> int:
        """Return total number of items including prompt variations."""
        if self.prompts:
            return len(self.items) * len(self.prompts)
        return len(self.items)
