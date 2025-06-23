from enum import Enum
import random
import json
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from src.datasets.base import BaseDataset
from pydantic import BaseModel


def map_grouping(grouping: str) -> str:
    """Map grouping to a more general category."""
    grouping = grouping.lower()

    if grouping == "black people":
        return "black"
    elif grouping == "muslims and arabic/middle eastern people":
        return "muslims/arabic"
    elif grouping == "asian people":
        return "asian"
    elif grouping == "latinx people":
        return "latino/hispanic"
    elif grouping == "white people":
        return "white"
    elif grouping == "lgbtq+ people":
        return "lgbtq+"
    else:
        return grouping


class isHateSpeech(str, Enum):
    true = "true"
    false = "false"


# Identity categories (specific groups)
class IdentityTargetCategory(str, Enum):
    women = "women"
    black = "black"
    lgbtq = "lgbtq+"
    muslims = "muslims/arabic"
    asian = "asian"
    hispanic = "latino/hispanic"
    jews = "jews"
    white = "white people"
    men = "men"
    christians = "christians"
    none = "none"


class IdentityContentClassification(BaseModel):
    is_hate_speech: isHateSpeech
    target_category: IdentityTargetCategory


class YoderIdentityDataset(BaseDataset):

    def __init__(
        self,
        data_path: str,
        tokenizer,
        prompts_file: str,
        max_samples: Optional[int] = None,
        seed: int = 42,
        text_field: str = "text",
        is_hate_field: str = "hate",
        target_field: str = "grouping",
        fold: Optional[str] = None,
        **additional_params: Any
    ):
        self.tokenizer = tokenizer
        self.prompts_file = prompts_file
        self.prompts = {}
        self.persona_ids = []
        self.data_df = None
        self.text_field = text_field
        self.is_hate_field = is_hate_field
        self.target_field = target_field
        self.fold = fold
        self._load_prompts()

        super().__init__(data_path, max_samples, seed, **additional_params)

    def _load_prompts(self) -> None:
        prompts_df = pd.read_parquet(self.prompts_file)
        for _, row in prompts_df.iterrows():
            self.prompts[row["persona_id"]] = (row["prompt"], row["persona_pos"])
        self.persona_ids = list(self.prompts.keys())
       

    def load_dataset(self) -> None:

        with open(self.data_path, "r") as f:
            data = [json.loads(line) for line in f]

        if self.fold:
            data = [item for item in data if item.get("fold") == self.fold]

        if self.max_samples:
            random.seed(self.seed)
            data = random.sample(data, min(self.max_samples, len(data)))

        self.data_df = pd.DataFrame(data)

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, str, Dict]:
        num_prompts = len(self.prompts)
        item_idx = idx // num_prompts
        prompt_idx = idx % num_prompts

        item = self.data_df.iloc[item_idx].to_dict()
        label = {
            "hate": item[self.is_hate_field],
            "target": map_grouping(item.get(self.target_field, "none")),
        }
        persona_id = self.persona_ids[prompt_idx]
        prompt, persona_pos = self.prompts[persona_id]

        prompt = prompt.replace("[TEXT]", item[self.text_field])

        # Prepare message
        message = {"role": "user", "content": prompt}

        # Process with model tokenizer
        prompt_text = self.tokenizer.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=True
        )

        return prompt_text, label, str(item_idx), persona_id, persona_pos

    def __len__(self) -> int:
        return self.data_df.shape[0] * len(self.prompts)
