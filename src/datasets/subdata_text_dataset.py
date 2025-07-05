from enum import Enum
import random
from typing import Any, Dict, Optional, Tuple
import pandas as pd
from src.datasets.base import BaseDataset
from pydantic import BaseModel



class isHateSpeech(str, Enum):
    true = "true"
    false = "false"


# class SpecificTarget(str, Enum):
#     men = "men"
#     non_binary = "non_binary"
#     transgender = "transgender"
#     women = "women"
#     arabs = "arabs"
#     brits = "brits"
#     chinese = "chinese"
#     eastern_european = "eastern_european"
#     indians = "indians"
#     mexicans = "mexicans"
#     middle_eastern = "middle_eastern"
#     pakistani = "pakistani"
#     polish = "polish"
#     russians = "russians"
#     leftwingers = "leftwingers"
#     rightwingers = "rightwingers"
#     asians = "asians"
#     blacks = "blacks"
#     middle_aged = "middle_aged"
#     none = "none"


class ContentClassification(
    BaseModel
):  # Careful this is not my BaseModel, but the one from pydantic
    is_hate_speech: isHateSpeech
    # specific_target: SpecificTarget


class SubdataTextDataset(BaseDataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        prompts_file: str,
        max_samples: Optional[int] = None,
        extreme_pos_personas_path: Optional[str] = None,
        prompt_template: Optional[str] = None,
        seed: int = 42,
        text_field: str = "text",
        split: Optional[str] = None,
        **additional_params: Any,
    ):
        self.tokenizer = tokenizer
        self.prompts = {}
        self.persona_ids = []
        self.data_df = None
        self.text_field = text_field
        self.split = split

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
        """Load text dataset from CSV."""
        df = pd.read_csv(self.data_path)

        # Apply sampling if needed
        if self.max_samples:
            random.seed(self.seed)
            df = df.sample(n=min(self.max_samples, len(df)), random_state=self.seed)

        if self.split:
            df = df[df["category"] == self.split]

        # set every label to {is_hate_speech: True}
        df["labels"] = df.apply( # TODO to check
            lambda x: {
                "is_hate_speech": "true", 
                "target_category": x["target"]
                }, axis=1
            )
        df["item_id"] = df.index

        self.data_df = df

        # for idx, row in df.iterrows():
        #     item_data = {
        #         "text": row[self.text_field],
        #         "labels": {
        #             "is_hate_speech": "true",  # All items in this dataset are hate speech
        #             "target_category": row.get("category", "none"),
        #             "specific_target": row.get("target", "none"),
        #         },
        #         "item_id": f"text_{idx}",
        #     }
        #     self.items.append(item_data)

        # self.data_df = pd.DataFrame(self.items)

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, str, Dict]:
        num_prompts = len(self.prompts)
        item_idx = idx // num_prompts
        prompt_idx = idx % num_prompts

        item = self.data_df.iloc[item_idx]
        persona_id = self.persona_ids[prompt_idx]
        prompt, persona_pos = self.prompts[persona_id]

        prompt = prompt.replace("[TEXT]", item[self.text_field])

        # Prepare message
        message = {"role": "user", "content": prompt}

        # Process with model tokenizer
        prompt_text = self.tokenizer.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=True
        )

        return prompt_text, item["labels"], item["item_id"], persona_id, persona_pos

    def __len__(self) -> int:
        return self.data_df.shape[0] * len(self.prompts)
    
    def convert_true_label(self, raw_label):
        return raw_label

