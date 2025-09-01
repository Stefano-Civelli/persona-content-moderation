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


class BinaryContentClassification(
    BaseModel
):  # Careful this is not my BaseModel, but the one from pydantic
    is_hate_speech: isHateSpeech



class SubdataTextDataset(BaseDataset):
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
        split: Optional[str] = None,
        **additional_params: Any,
    ):
        self.tokenizer = tokenizer
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

        if self.data_path.endswith(".tsv"):
            df = pd.read_csv(self.data_path, sep="\t")
        else:
            df = pd.read_csv(self.data_path)

        # if there is no category it means that we are using one of the specific category datasets
        # Likely we are using political_complete.csv
        if self.split:
            if "category" in df.columns:
                df = df[df["category"] == self.split]
            else:
                print(
                    "Not using the specified split because the given dataset is already of a specific category"
                )

        # Apply sampling if needed
        if self.max_samples:
            random.seed(self.seed)
            df = df.sample(n=min(self.max_samples, len(df)), random_state=self.seed)

        # set every label to {is_hate_speech: True}
        df["labels"] = df.apply(
            lambda x: {"is_hate_speech": True, "target_category": x["target"]}, axis=1
        )

        self.data_df = df

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, str, Dict]:
        num_texts = self.data_df.shape[0]
        prompt_idx = idx // num_texts
        item_idx = idx % num_texts

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

        return prompt_text, item["labels"], str(item["ID"]), persona_id, persona_pos

    def __len__(self) -> int:
        return self.data_df.shape[0] * len(self.prompts)

    def convert_true_label(self, raw_label):
        return raw_label
