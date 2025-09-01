import json
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple
import random
from pydantic import BaseModel
from enum import Enum
import pandas as pd
from src.datasets.base import BaseDataset


class MultiOffMemesDataset(BaseDataset):
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
        self.data_df = None

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
        labels_paths = [
            Path(self.data_path) / location
            for location in self.labels_relative_location
        ]
        img_relative_location =  Path(self.data_path.replace("interim", "raw")) / "Labelled Images"
        df = pd.DataFrame()
        
        for labels_path in labels_paths:
            if labels_path.exists():
                df = pd.concat([df, pd.read_csv(labels_path)], ignore_index=True)
            else:
                print(f"Labels file {labels_path} does not exist. Skipping.")
        # keep only rows where target is "Trump" or "Hillary"
        df = df[df["target"].isin(["Trump", "Hillary"])]
        print("Keeping only rows with target 'Trump' or 'Hillary'")

        if self.max_samples:
            random.seed(self.seed)
            df = df.sample(n=min(self.max_samples, len(df)), random_state=self.seed)

        df["labels"] = df.apply(
            lambda row: {"is_hate_speech": row["label"] == "offensive", "target": row["target"]}, axis=1
        )

        # add img_relative_location before the image name in the "image_name" column
        df["image_path"] = df["image_name"].apply(
            lambda img_name: str(Path(img_relative_location) / img_name)
        )

        self.data_df = df

        print(f"loaded a total of {self.data_df.shape[0]} images")

    def __getitem__(self, idx: int) -> Tuple[str, Image.Image, Dict, str, str, str]:

        num_images = self.data_df.shape[0]
        prompt_idx = idx // num_images
        item_idx = idx % num_images

        item = self.data_df.iloc[item_idx]
        persona_id = self.persona_ids[prompt_idx]
        unformatted_prompt, persona_pos = self.prompts[persona_id]

        # Load image as a PIL object
        image = Image.open(item["image_path"]).convert("RGB")
        max_size = (1024, 1024)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Return the raw components. The model class will format them.
        return (
            unformatted_prompt,
            image,
            item["labels"],
            str(item["image_name"].replace(".png", "").replace(".jpg", "")),
            persona_id,
            persona_pos,
        )

    def __len__(self) -> int:
        """Return total number of items including prompt variations."""
        return self.data_df.shape[0] * len(self.prompts)
       

    def convert_true_label(self, raw_label):
        return raw_label