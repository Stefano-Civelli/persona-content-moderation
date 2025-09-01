from enum import Enum
import json
from typing import Any, Dict, Optional, Tuple
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
# The one outputted by the model is the string on the right
class IdentityTargetCategory(str, Enum):
    women = "women"
    black = "black"
    lgbtq = "lgbtq+"
    muslims = "muslims/arabic"
    asian = "asian"
    hispanic = "latino/hispanic"
    jews = "jews"
    white = "white"
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
        prompts_file: Optional[str] = None,
        max_samples: Optional[int] = None,
        extreme_pos_personas_path: Optional[str] = None,
        prompt_template: Optional[str] = None,
        seed: int = 42,
        text_field: str = "text",
        is_hate_field: str = "hate",
        target_field: str = "grouping",
        fold: Optional[str] = None,
        target_group_size: Optional[int] = None,
        **additional_params: Any,
    ):
        self.tokenizer = tokenizer
        self.persona_ids = []
        self.data_df = None
        self.text_field = text_field
        self.is_hate_field = is_hate_field
        self.target_field = target_field
        self.fold = fold
        self.target_group_size = target_group_size

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

        with open(self.data_path, "r") as f:
            data = [json.loads(line) for line in f]

        self.data_df = pd.DataFrame(data)

        size_before_deduplication = self.data_df.shape[0]

        # eliminate duplicated rows using fold and text columns
        self.data_df = self.data_df.drop_duplicates(
            subset=["fold", self.text_field], keep="first"
        )
        print(
            f"Removed {size_before_deduplication - self.data_df.shape[0]} duplicate rows."
        )

        if self.fold:
            self.data_df = self.data_df[self.data_df["fold"] == self.fold]

        if self.target_group_size:
            self.data_df = self.data_df[
                self.data_df["target_groups"].apply(
                    lambda x: len(x) <= self.target_group_size
                )
            ]

        if self.max_samples:
            self.data_df = self.data_df.sample(
                n=self.max_samples, random_state=self.seed, replace=False
            )

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, str, Dict]:
        #I changed the order of items and personas, now its's much faster
        num_texts = self.data_df.shape[0]
        prompt_idx = idx // num_texts
        item_idx = idx % num_texts
        

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

        return prompt_text, label, str(item["ID"]), persona_id, persona_pos

    def __len__(self) -> int:
        return self.data_df.shape[0] * len(self.prompts)
    

    def convert_true_label(self, raw_label):
        # Dataset label format: {"hate": "yes"/"no", "target": "category_string"}
        # Output format for evaluation: {"is_hate_speech": "yes/no", "target_category": "category_string"}
        return {
            "is_hate_speech": raw_label.get("hate", isHateSpeech.false.value),
            "target_category": raw_label.get("target", IdentityTargetCategory.none.value),
        }

