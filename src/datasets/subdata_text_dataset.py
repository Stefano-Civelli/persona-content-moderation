from enum import Enum
import random
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from src.datasets.base import BaseDataset, DatasetConfig
from pydantic import BaseModel # Careful this is not my BaseModel, but the one from pydantic

class isHateSpeech(str, Enum):
    yes = "yes"
    no = "no"


class TargetCategory(str, Enum):
    age = "age"
    disabled = "disabled"
    gender = "gender"
    migration = "migration"
    origin = "origin"
    political = "political"
    race = "race"
    religion = "religion"
    sexuality = "sexuality"
    none = "none"


class SpecificTarget(str, Enum):
    men = "men"
    non_binary = "non_binary"
    transgender = "transgender"
    women = "women"
    arabs = "arabs"
    brits = "brits"
    chinese = "chinese"
    eastern_european = "eastern_european"
    indians = "indians"
    mexicans = "mexicans"
    middle_eastern = "middle_eastern"
    pakistani = "pakistani"
    polish = "polish"
    russians = "russians"
    leftwingers = "leftwingers"
    rightwingers = "rightwingers"
    asians = "asians"
    blacks = "blacks"
    middle_aged = "middle_aged"
    none = "none"


class ContentClassification(BaseModel): # Careful this is not my BaseModel, but the one from pydantic
    is_hate_speech: isHateSpeech
    target_category: TargetCategory
    specific_target: SpecificTarget


class HateSpeechTextDataset(BaseDataset):
    def __init__(
        self, 
        config: DatasetConfig,
        processor,  # Tokenizer for text models
        prompts_file: Optional[str] = None
    ):
        self.processor = processor
        self.prompts_file = prompts_file
        self.prompts = {}
        self.persona_ids = []
        
        if prompts_file:
            self._load_prompts()
        
        super().__init__(config)
        
        # Calculate total items similar to vision dataset
        if self.prompts:
            self.total_items = len(self.items) * len(self.prompts)
        else:
            self.total_items = len(self.items)
    
    def _load_prompts(self) -> None:
        """Load prompts from parquet file (similar to vision dataset)."""
        prompts_df = pd.read_parquet(self.prompts_file)
        
        # Create a dictionary with persona_id as key and prompt as value
        for _, row in prompts_df.iterrows():
            self.prompts[row['persona_id']] = (row['prompt'], row.get('persona_pos', 'neutral'))
        self.persona_ids = list(self.prompts.keys())
    
    def load_dataset(self) -> None:
        """Load text dataset from CSV."""
        df = pd.read_csv(self.data_path)
        
        # Apply sampling if needed
        if self.max_samples:
            random.seed(self.seed)
            df = df.sample(n=min(self.max_samples, len(df)), random_state=self.seed)
        
        for idx, row in df.iterrows():
            item_data = {
                'text': row['text'],
                'labels': {
                    'is_hate_speech': 'yes',  # All items in this dataset are hate speech
                    'target_category': row.get('category', 'none'),
                    'specific_target': row.get('target', 'none')
                },
                'item_id': f"text_{idx}"
            }
            self.items.append(item_data)
    
    def __len__(self) -> int:
        return self.total_items
    
    def __getitem__(self, idx: int) -> Tuple[Dict, Dict[str, Any], str, Dict[str, Any]]:
        """Get item with optional prompt variation (similar to vision dataset)."""
        if self.prompts:
            # Multi-prompt mode
            num_prompts = len(self.prompts)
            item_idx = idx // num_prompts
            prompt_idx = idx % num_prompts
            
            item = self.items[item_idx]
            persona_id = self.persona_ids[prompt_idx]
            prompt_template, persona_pos = self.prompts[persona_id]
            
            # Format the prompt with the text
            prompt = prompt_template.format(text=item['text'])
            
            metadata = {
                'persona_id': persona_id,
                'persona_pos': persona_pos,
                'text': item['text']
            }
        else:
            # Single prompt mode - use a default prompt
            item = self.items[idx]
            prompt = f"Classify the following text for hate speech: {item['text']}"
            metadata = {'text': item['text']}
        
        # Prepare message in chat format
        message = {
            "role": "user",
            "content": prompt
        }
        
        # Apply chat template
        prompt_text = self.processor.apply_chat_template(
            [message], 
            tokenize=False,
            add_generation_prompt=True
        )
        
        # For text models, we return the prompt as a simple dict
        inputs = {"prompt": prompt_text}
        
        return inputs, item['labels'], item['item_id'], metadata


def text_collate_fn(
    batch: List[Tuple[str, Dict, str, Dict]],
) -> Tuple[List[str], List[Dict], List[str], List[Dict]]:
    """Custom collate function for text data."""
    prompts, labels, ids, metadata = zip(*batch)
    return list(prompts), list(labels), list(ids), list(metadata)
