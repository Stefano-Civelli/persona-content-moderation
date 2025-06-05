import os
import json
import pickle
import random
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
import torch


from models.qwen import process_vision_info

def custom_collate_fn(batch):
    """
    Custom collate function for IDEFICS model with left padding
    
    Args:
        batch: List of tuples (inputs, labels, img_name)
    
    Returns:
        Tuple of (batched_inputs, batched_labels, img_names)
    """
    # Unzip the batch into separate lists
    inputs, labels, img_names, persona_ids, persona_pos = zip(*batch)
    
    # Process inputs
    pixel_values = [item['pixel_values'] for item in inputs]
    pixel_attention_mask = [item['pixel_attention_mask'] for item in inputs]
    input_ids = [item['input_ids'] for item in inputs]
    attention_mask = [item['attention_mask'] for item in inputs]
    
    #print(f'Pixel values shape: {pixel_values[0].shape}')
    # Stack pixel values and pixel attention mask
    pixel_values = torch.stack(pixel_values)
    pixel_attention_mask = torch.stack(pixel_attention_mask)
    
    # Find max length in this batch
    max_len = max(ids.size(0) for ids in input_ids)
    batch_size = len(input_ids)
    
    # Create tensors for left-padded sequences
    padded_input_ids = torch.full((batch_size, max_len), 128002, dtype=input_ids[0].dtype)
    padded_attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask[0].dtype)
    
    # Fill in the sequences from the right
    for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
        seq_len = ids.size(0)
        padded_input_ids[i, -seq_len:] = ids
        padded_attention_mask[i, -seq_len:] = mask
    
    # Create batched inputs dictionary
    batched_inputs = {
        'pixel_values': pixel_values,
        'pixel_attention_mask': pixel_attention_mask,
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask
    }
    
    # Process labels - assuming they're dictionaries
    batched_labels = {}
    if labels and isinstance(labels[0], dict):
        all_keys = set().union(*[d.keys() for d in labels])
        
        for key in all_keys:
            values = [d.get(key) for d in labels]
            if all(isinstance(v, (int, float, bool, torch.Tensor)) for v in values):
                if isinstance(values[0], torch.Tensor):
                    batched_labels[key] = torch.stack(values)
                else:
                    batched_labels[key] = torch.tensor(values)
            else:
                batched_labels[key] = values
    
    return batched_inputs, batched_labels, list(img_names), list(persona_ids), list(persona_pos)



def custom_collate_fn_no_personas(batch):
    # Unzip the batch into separate lists
    inputs, labels, img_names = zip(*batch)
    
    # Process inputs
    pixel_values = [item['pixel_values'] for item in inputs]
    pixel_attention_mask = [item['pixel_attention_mask'] for item in inputs]
    input_ids = [item['input_ids'] for item in inputs]
    attention_mask = [item['attention_mask'] for item in inputs]
    
    #print(f'Pixel values shape: {pixel_values[0].shape}')
    # Stack pixel values and pixel attention mask
    pixel_values = torch.stack(pixel_values)
    pixel_attention_mask = torch.stack(pixel_attention_mask)
    
    # Find max length in this batch
    max_len = max(ids.size(0) for ids in input_ids)
    batch_size = len(input_ids)
    
    # Create tensors for left-padded sequences
    padded_input_ids = torch.full((batch_size, max_len), 128002, dtype=input_ids[0].dtype)
    padded_attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask[0].dtype)
    
    # Fill in the sequences from the right
    for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
        seq_len = ids.size(0)
        padded_input_ids[i, -seq_len:] = ids
        padded_attention_mask[i, -seq_len:] = mask
    
    # Create batched inputs dictionary
    batched_inputs = {
        'pixel_values': pixel_values,
        'pixel_attention_mask': pixel_attention_mask,
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask
    }
    
    # Process labels - assuming they're dictionaries
    batched_labels = {}
    if labels and isinstance(labels[0], dict):
        all_keys = set().union(*[d.keys() for d in labels])
        
        for key in all_keys:
            values = [d.get(key) for d in labels]
            if all(isinstance(v, (int, float, bool, torch.Tensor)) for v in values):
                if isinstance(values[0], torch.Tensor):
                    batched_labels[key] = torch.stack(values)
                else:
                    batched_labels[key] = torch.tensor(values)
            else:
                batched_labels[key] = values
    
    return batched_inputs, batched_labels, list(img_names)





def custom_collate_fn_img(batch):
    """
    Custom collate function for IDEFICS model with left padding for text and right padding for images
    """
    # Unzip the batch into separate lists
    inputs, labels, img_names = zip(*batch)
    
    # Process inputs
    pixel_values = [item['pixel_values'] for item in inputs]
    pixel_attention_mask = [item['pixel_attention_mask'] for item in inputs]
    input_ids = [item['input_ids'] for item in inputs]
    attention_mask = [item['attention_mask'] for item in inputs]
    
    # Get dimensions from first image
    num_patches, channels, height, width = pixel_values[0].shape
    
    # Find max dimensions for images in the batch
    max_height = max(p.shape[-2] for p in pixel_values)
    max_width = max(p.shape[-1] for p in pixel_values)
    
    # Create padded pixel values and attention masks
    batch_size = len(pixel_values)
    padded_pixel_values = torch.zeros((batch_size, num_patches, channels, max_height, max_width), 
                                    dtype=pixel_values[0].dtype)
    padded_pixel_attention_mask = torch.zeros((batch_size, num_patches, max_height, max_width), 
                                            dtype=pixel_attention_mask[0].dtype)
    
    # Fill in the images and their attention masks
    for i, (img, mask) in enumerate(zip(pixel_values, pixel_attention_mask)):
        h, w = img.shape[-2:]
        padded_pixel_values[i, :, :, :h, :w] = img
        padded_pixel_attention_mask[i, :, :h, :w] = mask
    
    # Find max length for text sequences in this batch
    max_len = max(ids.size(0) for ids in input_ids)
    
    # Create tensors for left-padded sequences
    padded_input_ids = torch.full((batch_size, max_len), 128002, dtype=input_ids[0].dtype)
    padded_attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask[0].dtype)
    
    # Fill in the sequences from the right
    for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
        seq_len = ids.size(0)
        padded_input_ids[i, -seq_len:] = ids
        padded_attention_mask[i, -seq_len:] = mask
    
    # Create batched inputs dictionary
    batched_inputs = {
        'pixel_values': padded_pixel_values,
        'pixel_attention_mask': padded_pixel_attention_mask,
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask
    }
    
    # Process labels - assuming they're dictionaries
    batched_labels = {}
    if labels and isinstance(labels[0], dict):
        all_keys = set().union(*[d.keys() for d in labels])
        
        for key in all_keys:
            values = [d.get(key) for d in labels]
            if all(isinstance(v, (int, float, bool, torch.Tensor)) for v in values):
                if isinstance(values[0], torch.Tensor):
                    batched_labels[key] = torch.stack(values)
                else:
                    batched_labels[key] = torch.tensor(values)
            else:
                batched_labels[key] = values
    
    return batched_inputs, batched_labels, list(img_names)

def qwen_custom_collate_fn(batch):
    # Unzip the batch into separate lists
    inputs, labels, img_names, persona_ids, persona_pos = zip(*batch)
    
    # Find max length for input_ids in this batch
    max_len = max(item['input_ids'].size(0) for item in inputs)
    batch_size = len(inputs)
    
    # Initialize tensors for left-padded sequences
    padded_input_ids = torch.full((batch_size, max_len), 151643, dtype=torch.long)  # Qwen padding token
    padded_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    
    # Process each item in the batch
    image_list = []
    for i, item in enumerate(inputs):
        # Get current sequence
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        seq_len = input_ids.size(0)
        
        # Fill in sequences from the right (left padding)
        padded_input_ids[i, -seq_len:] = input_ids
        padded_attention_mask[i, -seq_len:] = attention_mask
        
        # Handle image inputs
        if 'image' in item:
            image_list.append(item['image'])
    
    # Create batched inputs dictionary
    batched_inputs = {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
    }
    
    # Add image tensor if present
    if image_list:
        batched_inputs['image'] = torch.stack(image_list)
    
    # Process labels - assuming they're dictionaries
    batched_labels = {}
    if labels and isinstance(labels[0], dict):
        all_keys = set().union(*[d.keys() for d in labels])
        
        for key in all_keys:
            values = [d.get(key) for d in labels]
            if all(isinstance(v, (int, float, bool, torch.Tensor)) for v in values):
                if isinstance(values[0], torch.Tensor):
                    batched_labels[key] = torch.stack(values)
                else:
                    batched_labels[key] = torch.tensor(values)
            else:
                batched_labels[key] = values
    
    return batched_inputs, batched_labels, list(img_names), list(persona_ids), list(persona_pos)

class MemeDataset(Dataset):
    def __init__(self, base_path: str, processor, max_samples: Optional[int] = None):
        self.base_path = base_path
        self.processor = processor
        self.images: List[str] = []
        self.labels: List[int] = []
        self.load_dataset(max_samples)

    def load_dataset(self, max_samples: Optional[int]) -> None:
        """Load dataset metadata from JSONL file"""
        train_path = os.path.join(self.base_path, 'train.jsonl')
        self.images_path = os.path.join(self.base_path, 'img')
        
        with open(train_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        if max_samples:
            data = data[:max_samples]
            
        for item in data:
            img_path = os.path.join(self.images_path, os.path.basename(item['img']))
            if os.path.exists(img_path):
                self.images.append(img_path)
                self.labels.append(item['label'])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Dict, int]:
        """Get a single item from the dataset"""
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        # Create prompt for the image
        message = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Is this meme harmful or not harmful? Please respond with exactly one word: either 'harmful' or 'not_harmful'."},
            ]
        }
        
        prompt = self.processor.apply_chat_template([message], add_generation_prompt=True)
        
        # Process the image
        inputs = self.processor(
            text=prompt,
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # Remove the batch dimension added by the processor
        # TODO Maybe this is not necessary
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        return inputs, label
        
## Version 2


class MultiPromptMemeDataset(Dataset):
    def __init__(self, base_path: str, processor, prompts_file: str, max_samples: Optional[int] = None):
        """
        Initialize the dataset with multiple prompts per image.
        
        Args:
            base_path: Base path containing the dataset
            processor: The processor for preparing inputs
            prompts_file: Path to the file containing classification prompts
            max_samples: Optional maximum number of image samples to load
        """
        self.base_path = base_path
        self.processor = processor
        self.prompts_file = prompts_file
        self.images: List[str] = []
        self.labels: List[int] = []
        self.prompts: List[str] = []
        
        # Load dataset and prompts
        self.load_dataset(max_samples)
        self.load_prompts()
        
        # Calculate total number of items (images × prompts)
        self.total_items = len(self.images) * len(self.prompts)

    def load_prompts(self) -> None:
        """Load classification prompts from file"""
        self.prompts = pd.read_parquet(self.prompts_file)['prompt'].tolist()

            
    def load_dataset(self, max_samples: Optional[int]) -> None:
        """Load dataset metadata from JSONL file"""
        train_path = os.path.join(self.base_path, 'train.jsonl')
        self.images_path = os.path.join(self.base_path, 'img')


        
        with open(train_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        if max_samples:
            # sample randomly from the dataset if max_samples is set
            # set a random seed for reproducibility
            random.seed(42)
            data = random.sample(data, max_samples)
            
            
        for item in data:
            img_path = os.path.join(self.images_path, os.path.basename(item['img']))
            if os.path.exists(img_path):
                self.images.append(img_path)
                self.labels.append(item['label'])

    def __len__(self) -> int:
        """Return total number of items (images × prompts)"""
        return self.total_items

    def __getitem__(self, idx: int) -> Tuple[Dict, int]:
        """
        Get a single item from the dataset.
        
        The index is converted into an (image_idx, prompt_idx) pair:
        - image_idx = idx // num_prompts
        - prompt_idx = idx % num_prompts
        
        Returns:
            Tuple containing processed inputs and label
        """
        num_prompts = len(self.prompts)
        image_idx = idx // num_prompts
        prompt_idx = idx % num_prompts
        
        
        image = Image.open(self.images[image_idx]).convert('RGB')
        label = self.labels[image_idx]
        prompt = self.prompts[prompt_idx]
        
        # Create prompt for the image using the selected prompt
        message = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]
        }
        
        prompt = self.processor.apply_chat_template([message], add_generation_prompt=True)
        
        # Process the image
        inputs = self.processor(
            text=prompt,
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # Remove the batch dimension added by the processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}


        print(f'Dataloader: {inputs}, {label}, {self.images[image_idx]}')
        return inputs, label, self.images[image_idx]

    def get_prompt_index(self, idx: int) -> int:
        """Helper method to get the prompt index for a given dataset index"""
        return idx % len(self.prompts)
    
    def get_image_index(self, idx: int) -> int:
        """Helper method to get the image index for a given dataset index"""
        return idx // len(self.prompts)
    
    def get_prompt_text(self, idx: int) -> str:
        """Helper method to get the prompt text for a given dataset index"""
        prompt_idx = self.get_prompt_index(idx)
        return self.prompts[prompt_idx]


class MultiPromptMultiLabelMemeDataset(Dataset):
    def __init__(self, base_path: str, processor, prompts_file: str, max_samples: Optional[int] = None):
        self.base_path = base_path
        self.processor = processor
        self.prompts_file = prompts_file
        self.images: List[str] = []
        self.labels: List[Dict[str, List[str]]] = []  # Changed to store gold labels
        self.prompts, self.persona_ids = self.load_prompts()
        
        self.load_dataset(max_samples)
        self.total_items = len(self.images) * len(self.prompts)

    def load_prompts(self) -> Dict:
        prompts_df = pd.read_parquet(self.prompts_file)

        # create a dictionary with persona_id as key and prompt as value
        prompts = {}
        for _, row in prompts_df.iterrows():
            prompts[row['persona_id']] = (row['prompt'], row['persona_pos'])

        persona_ids = list(prompts.keys())

        return prompts, persona_ids
            
    def load_dataset(self, max_samples: Optional[int]) -> None:
        train_path = os.path.join(self.base_path, 'fine_grained_labels/train.jsonl')
        self.images_path = os.path.join(self.base_path)

        with open(train_path, 'r') as f:
            opened_data = [json.loads(line) for line in f]

        # filter away data items where gold_pc or gold_attack contain more than one label
        data = [item for item in opened_data if len(item['gold_pc']) == 1 and len(item['gold_attack']) == 1]

        print(f'Number of filtered items: {len(opened_data) - len(data)}')
        
        if max_samples:
            # keep only hateful images
            # data = [item for item in data if item['gold_hate'][0] == 'hateful']
            # Sample randomly from the dataset if max_samples is set
            random.seed(42)  # Set seed for reproducibility
            data = random.sample(data, max_samples)
            
        for item in data:
            img_path = os.path.join(self.images_path, item['img'])
            if os.path.exists(img_path):
                self.images.append(img_path)
                
                # Store gold labels
                labels = {
                    'hate': item['gold_hate'],
                    'pc': item['gold_pc'],
                    'attack': item['gold_attack']
                }
                self.labels.append(labels)

    def __len__(self) -> int:
        return self.total_items

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict[str, List[str]], str]:
        num_prompts = len(self.prompts)
        #num_images = len(self.images)
        image_idx = idx // num_prompts
        prompt_idx = idx % num_prompts
        
        # Load and process image
        image = Image.open(self.images[image_idx]).convert('RGB')

        label = self.labels[image_idx]
        persona_id = self.persona_ids[prompt_idx]
        prompt = self.prompts[persona_id][0]
        persona_pos = self.prompts[persona_id][1]
        
        message = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]
        }
        
        prompt = self.processor.apply_chat_template([message], add_generation_prompt=True)
        
        inputs = self.processor(
            text=prompt,
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # assert that the attention mask is all 1s
        #print(self.images[image_idx])
        #assert torch.all(inputs['attention_mask'] == 1)
        return inputs, label, self.images[image_idx], persona_id, persona_pos
    

class MultiPromptMultiLabelMemeDatasetTestSet(Dataset):
    def __init__(self, base_path: str, processor, prompts_file: str, max_samples: Optional[int] = None):
        self.base_path = base_path
        self.processor = processor
        self.prompts_file = prompts_file
        self.images: List[str] = []
        self.prompts = self.load_prompts()
        
        self.load_dataset(max_samples)
        self.total_items = len(self.images) * len(self.prompts)

    def load_prompts(self) -> Dict:
        prompts_df = pd.read_parquet(self.prompts_file)

        # create a dictionary with persona_id as key and prompt as value
        prompts = {}
        for _, row in prompts_df.iterrows():
            prompts[row['persona_id']] = (row['prompt'], row['persona_pos'])

        return prompts
            
    def load_dataset(self, max_samples: Optional[int]) -> None:
        test_path = os.path.join(self.base_path, 'fine_grained_labels/test.jsonl')
        self.images_path = os.path.join(self.base_path)


        with open(test_path, 'r') as f:
            data = [json.loads(line) for line in f]

        if max_samples:
            # Sample randomly from the dataset if max_samples is set
            random.seed(42)  # Set seed for reproducibility
            data = random.sample(data, max_samples)

        for item in data:
            img_path = os.path.join(self.images_path, item['img'])
            if os.path.exists(img_path):
                self.images.append(img_path)
                
            

    def __len__(self) -> int:
        return self.total_items

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict[str, List[str]], str]:
        num_prompts = len(self.prompts)
        image_idx = idx // num_prompts
        prompt_idx = idx % num_prompts
        
        # Load and process image
        image = Image.open(self.images[image_idx]).convert('RGB')

        persona_id = list(self.prompts.keys())[prompt_idx]
        prompt = self.prompts[persona_id][0]
        persona_pos = self.prompts[persona_id][1]
        
        message = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]
        }
        
        prompt = self.processor.apply_chat_template([message], add_generation_prompt=True)
        
        inputs = self.processor(
            text=prompt,
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        #print(f'Dataloader: {inputs}, {self.images[image_idx]}')
        return inputs, self.images[image_idx], persona_id, persona_pos

class MultiPromptMultiLabelMemeDatasetQwen(Dataset):
    def __init__(self, base_path: str, processor, prompts_file: str, max_samples: Optional[int] = None):
        # Keep existing initialization
        self.base_path = base_path
        self.processor = processor
        self.prompts_file = prompts_file
        self.images = []
        self.labels = []
        self.prompts = []
        
        # Load extreme positions
        with open('../results/extreme_positions_diagonal_15_per_quadrant.pkl', 'rb') as f:
            extreme_pos = pickle.load(f)
        self.extreme_pos_ids = [pos[0] for pos in extreme_pos['top_right'] + extreme_pos['top_left'] + 
                              extreme_pos['bottom_right'] + extreme_pos['bottom_left']]

        self.id_to_pos = {}
        for pos_name in ['top_right', 'top_left', 'bottom_right', 'bottom_left']:
            for pos in extreme_pos[pos_name]:
                self.id_to_pos[pos[0]] = pos_name
                
        self.load_dataset(max_samples)
        self.load_prompts()
        self.total_items = len(self.images) * len(self.prompts)

    def load_prompts(self) -> None:
        self.prompts = pd.read_parquet(self.prompts_file)['prompt'].tolist()
            
    def load_dataset(self, max_samples: Optional[int]) -> None:
        # Keep existing dataset loading logic
        train_path = os.path.join(self.base_path, 'fine_grained_labels/train.jsonl')
        self.images_path = os.path.join(self.base_path)

        with open(train_path, 'r') as f:
            opened_data = [json.loads(line) for line in f]

        data = [item for item in opened_data if len(item['gold_pc']) == 1 and len(item['gold_attack']) == 1]
        print(f'Number of filtered items: {len(opened_data) - len(data)}')
        
        if max_samples:
            random.seed(42)
            data = random.sample(data, max_samples)
            
        for item in data:
            img_path = os.path.join(self.images_path, item['img'])
            if os.path.exists(img_path):
                self.images.append(img_path)
                labels = {
                    'hate': item['gold_hate'],
                    'pc': item['gold_pc'],
                    'attack': item['gold_attack']
                }
                self.labels.append(labels)

    def __len__(self) -> int:
        return self.total_items

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict[str, List[str]], str]:
        num_prompts = len(self.prompts)
        image_idx = idx // num_prompts
        prompt_idx = idx % num_prompts
        
        # Load image
        image = Image.open(self.images[image_idx]).convert('RGB')

        label = self.labels[image_idx]
        prompt = self.prompts[prompt_idx]
        person_id = self.extreme_pos_ids[prompt_idx]
        person_pos = self.id_to_pos[person_id]
        
        # Create Qwen-specific message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process using Qwen's format
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, _ = process_vision_info(messages)
        
        # Create model inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        return inputs, label, self.images[image_idx], person_id, person_pos


class MultiPromptMultiLabelMMHS150KDataset(Dataset):
    def __init__(self, base_path: str, processor, prompts_file: str, max_samples: Optional[int] = None):
        """
        Initialize the dataset with multiple prompts per image.
        
        Args:
            base_path (str): Base path containing the dataset
            processor: Image/text processor
            prompts_file (str): Path to the prompts parquet file
            max_samples (Optional[int]): Maximum number of samples to load
        """
        self.base_path = base_path
        self.processor = processor
        self.prompts_file = prompts_file
        self.images: List[str] = []
        self.labels: List[Dict[str, List[str]]] = []
        self.tweet_texts: List[str] = []  # New attribute for tweet texts
        self.tweet_ids: List[str] = []  # Store tweet IDs separately
        self.prompts = self.load_prompts()

        # with open('../results/extreme_positions_diagonal_15_per_quadrant.pkl', 'rb') as f:
        #     extreme_pos = pickle.load(f)
        # self.extreme_pos_ids = [pos[0] for pos in extreme_pos['top_right'] + extreme_pos['top_left'] + extreme_pos['bottom_right'] + extreme_pos['bottom_left']]

        # self.id_to_pos = {}
        # for pos_name in ['top_right', 'top_left', 'bottom_right', 'bottom_left']:
        #     for pos in extreme_pos[pos_name]:
        #         self.id_to_pos[pos[0]] = pos_name

        
        # Load dataset and prompts
        self.load_dataset(max_samples)
        
        # Calculate total number of items (images × prompts)
        self.total_items = len(self.images) * len(self.prompts)

    def load_prompts(self) -> Dict:
        prompts_df = pd.read_parquet(self.prompts_file)

        # create a dictionary with persona_id as key and prompt as value
        prompts = {}
        for _, row in prompts_df.iterrows():
            prompts[row['persona_id']] = (row['prompt'], row['persona_pos'])

        return prompts
            
    def load_dataset(self, max_samples: Optional[int]) -> None:
        """
        Load dataset metadata from JSON file
        
        Args:
            max_samples (Optional[int]): Maximum number of samples to load randomly
        """
        dataset_path = os.path.join(self.base_path, 'MMHS150K_GT.json')
        self.images_path = os.path.join(self.base_path, 'img_resized')  # Updated images path

        with open(dataset_path, 'r') as f:
            data = json.load(f)

        # Convert data from dictionary format to list of items
        processed_data = []
        for tweet_id, tweet_info in data.items():
            processed_data.append({
                'id': tweet_id,
                'tweet_text': tweet_info['tweet_text'],
                'labels_str': tweet_info['labels_str']
            })

        if max_samples:
            random.seed(42)  # Set seed for reproducibility
            processed_data = random.sample(processed_data, max_samples)
            
        for item in processed_data:
            # Construct image path using tweet_id
            img_path = os.path.join(self.images_path, f"{item['id']}.jpg")
            if os.path.exists(img_path):
                self.images.append(img_path)
                self.tweet_texts.append(item['tweet_text'])
                self.tweet_ids.append(item['id'])  # Store tweet ID
                
                # Store labels in the format expected by the model
                labels = {
                    'labels': item['labels_str']
                }
                self.labels.append(labels)

    def __len__(self) -> int:
        """Return total number of items (images × prompts)"""
        return self.total_items

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict[str, List[str]], str, str, str]:
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item to get
            
        Returns:
            Tuple containing:
            - processed_inputs (Dict): Processed inputs for the model
            - labels (Dict[str, List[str]]): Dictionary of gold labels
            - image_path (str): Path to the image
            - person_id (str): ID of the person
            - person_pos (str): Position of the person
        """
        num_prompts = len(self.prompts)
        image_idx = idx // num_prompts
        prompt_idx = idx % num_prompts
        
        # Load and process image
        image = Image.open(self.images[image_idx]).convert('RGB')

        label = self.labels[image_idx]
        persona_id = list(self.prompts.keys())[prompt_idx]
        prompt = self.prompts[persona_id][0]
        persona_pos = self.prompts[persona_id][1]
        tweet_text = self.tweet_texts[image_idx]
        
        # Replace [TWEET_TEXT] placeholder with actual tweet text
        tweet_text = tweet_text[:200]
        prompt = prompt.replace('[TWEET_TEXT]', tweet_text)
        # Create prompt for the image
        message = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]
        }
        
        prompt = self.processor.apply_chat_template([message], add_generation_prompt=True)
        
        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return inputs, label, self.images[image_idx], persona_id, persona_pos
    

def get_persona_ids(personas_file_path: str) -> List[str]:
    with open(personas_file_path, 'rb') as f:
            personas = pickle.load(f)
    
    persona_ids = [pos[0] for pos in personas['leftmost'] + personas['rightmost']]

    return persona_ids

def get_persona_id_to_pos_dict(personas_file_path: str) -> Dict[str, str]:
    with open(personas_file_path, 'rb') as f:
            personas = pickle.load(f)
    
    id_to_pos = {}
    for pos_name in ['leftmost', 'rightmost']:
        for pos in personas[pos_name]:
            id_to_pos[pos[0]] = pos_name

    return id_to_pos



def collate_fn_US_2016(batch):
    """
    Custom collate function for IDEFICS model with left padding
    
    Args:
        batch: List of tuples (inputs, labels, img_name)
    
    Returns:
        Tuple of (batched_inputs, batched_labels, img_names)
    """
    # Unzip the batch into separate lists
    inputs, img_names, persona_ids, persona_pos = zip(*batch)
    
    # Process inputs
    pixel_values = [item['pixel_values'] for item in inputs]
    pixel_attention_mask = [item['pixel_attention_mask'] for item in inputs]
    input_ids = [item['input_ids'] for item in inputs]
    attention_mask = [item['attention_mask'] for item in inputs]
    
    #print(f'Pixel values shape: {pixel_values[0].shape}')
    # Stack pixel values and pixel attention mask
    pixel_values = torch.stack(pixel_values)
    pixel_attention_mask = torch.stack(pixel_attention_mask)
    
    # Find max length in this batch
    max_len = max(ids.size(0) for ids in input_ids)
    batch_size = len(input_ids)
    
    # Create tensors for left-padded sequences
    padded_input_ids = torch.full((batch_size, max_len), 128002, dtype=input_ids[0].dtype)
    padded_attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask[0].dtype)
    
    # Fill in the sequences from the right
    for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
        seq_len = ids.size(0)
        padded_input_ids[i, -seq_len:] = ids
        padded_attention_mask[i, -seq_len:] = mask
    
    # Create batched inputs dictionary
    batched_inputs = {
        'pixel_values': pixel_values,
        'pixel_attention_mask': pixel_attention_mask,
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask
    }
    
    
    return batched_inputs, list(img_names), list(persona_ids), list(persona_pos)


class US2016Dataset(Dataset):
    def __init__(self, base_path: str, processor, prompts_file: str, max_samples: Optional[int] = None):
        self.base_path = base_path
        self.processor = processor
        self.prompts_file = prompts_file
        self.images: List[str] = []
        self.prompts: List[str] = []

        self.extreme_pos_ids = get_persona_ids('../results/left_and_right_40_personas.pkl')
        self.id_to_pos = get_persona_id_to_pos_dict('../results/left_and_right_40_personas.pkl')
        
        self.load_dataset(max_samples)
        self.load_prompts()
        self.total_items = len(self.images) * len(self.prompts)
        print(f'Total items: {self.total_items}')

    def load_prompts(self) -> None:
        self.prompts = pd.read_parquet(self.prompts_file)['prompt'].tolist()
        print(f'Number of prompts: {len(self.prompts)}')

    def load_dataset(self, max_samples: Optional[int]) -> None:
        data = pd.read_csv(os.path.join(self.base_path, 'custom_DEM_plus_REP.csv'))

        
        if max_samples:
            data = data.sample(n=max_samples, random_state=42)
            
        valid_images = []
        for _, row in data.iterrows():
            img_name = row['link'].split('/')[-1]
            png_path = os.path.join(self.base_path, 'images/imgur_US_2016', img_name + '.png')
            jpeg_path = os.path.join(self.base_path, 'images/imgur_US_2016', img_name + '.jpeg')
            
            if os.path.exists(png_path):
                valid_images.append(png_path)
            elif os.path.exists(jpeg_path):
                valid_images.append(jpeg_path)
        
        self.images = valid_images
        print(f'Number of images: {len(self.images)}')

    def __len__(self) -> int:
        return self.total_items

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict[str, List[str]], str]:
        num_prompts = len(self.prompts)
        image_idx = idx // num_prompts
        prompt_idx = idx % num_prompts
        
        img_path = self.images[image_idx]
        image = Image.open(img_path).convert('RGB')
        prompt = self.prompts[prompt_idx]
        person_id = self.extreme_pos_ids[prompt_idx]
        person_pos = self.id_to_pos[person_id]
        
        message = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]
        }
        
        prompt = self.processor.apply_chat_template([message], add_generation_prompt=True)
        
        inputs = self.processor(
            text=prompt,
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}


        return inputs, img_path, person_id, person_pos


class SinglePromptMemeDataset(Dataset):
    def __init__(self, base_path: str, processor, prompt: str, max_samples: Optional[int] = None):
        self.base_path = base_path
        self.processor = processor
        self.prompt = prompt
        self.images: List[str] = []
        self.labels: List[Dict[str, List[str]]] = []
        
        self.load_dataset(max_samples)
        self.total_items = len(self.images)  # Total items is just number of images now

    def load_dataset(self, max_samples: Optional[int]) -> None:
        """Load and filter the dataset."""
        train_path = os.path.join(self.base_path, 'fine_grained_labels/train.jsonl')
        self.images_path = os.path.join(self.base_path)

        with open(train_path, 'r') as f:
            opened_data = [json.loads(line) for line in f]

        # Filter away data items where gold_pc or gold_attack contain more than one label
        data = [item for item in opened_data if len(item['gold_pc']) == 1 and len(item['gold_attack']) == 1]

        print(f'Number of filtered items: {len(opened_data) - len(data)}')
        
        if max_samples:
            # Sample randomly from the dataset if max_samples is set
            random.seed(42)  # Set seed for reproducibility
            data = random.sample(data, max_samples)
            
        for item in data:
            img_path = os.path.join(self.images_path, item['img'])
            if os.path.exists(img_path):
                self.images.append(img_path)
                
                # Store gold labels
                labels = {
                    'hate': item['gold_hate'],
                    'pc': item['gold_pc'],
                    'attack': item['gold_attack']
                }
                self.labels.append(labels)

    def __len__(self) -> int:
        return self.total_items

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict[str, List[str]], str]:

        # Load and process image
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        
        # Create message with single prompt
        message = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": self.prompt},
            ]
        }
        
        # Process the inputs
        prompt = self.processor.apply_chat_template([message], add_generation_prompt=True)
        inputs = self.processor(
            text=prompt,
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        return inputs, label, self.images[idx]



class SinglePromptMMHS150KDataset(Dataset):
    def __init__(self, base_path: str, processor, prompt: str, max_samples: Optional[int] = None):

        self.base_path = base_path
        self.processor = processor
        self.prompt = prompt
        self.images: List[str] = []
        self.labels: List[Dict[str, List[str]]] = []
        self.tweet_texts: List[str] = []
        self.tweet_ids: List[str] = []
        
        # Load dataset
        self.load_dataset(max_samples)
        
        # Total items is just number of images since we have only one prompt
        self.total_items = len(self.images)

    def load_dataset(self, max_samples: Optional[int]) -> None:
        dataset_path = os.path.join(self.base_path, 'MMHS150K_GT.json')
        self.images_path = os.path.join(self.base_path, 'img_resized')

        with open(dataset_path, 'r') as f:
            data = json.load(f)

        # Convert data from dictionary format to list of items
        processed_data = []
        for tweet_id, tweet_info in data.items():
            processed_data.append({
                'id': tweet_id,
                'tweet_text': tweet_info['tweet_text'],
                'labels_str': tweet_info['labels_str']
            })

        if max_samples:
            random.seed(42)  # Set seed for reproducibility
            processed_data = random.sample(processed_data, max_samples)
            
        for item in processed_data:
            # Construct image path using tweet_id
            img_path = os.path.join(self.images_path, f"{item['id']}.jpg")
            if os.path.exists(img_path):
                self.images.append(img_path)
                self.tweet_texts.append(item['tweet_text'])
                self.tweet_ids.append(item['id'])
                
                # Store labels in the format expected by the model
                labels = {
                    'labels': item['labels_str']
                }
                self.labels.append(labels)

    def __len__(self) -> int:
        """Return total number of items (just number of images now)"""
        return self.total_items

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict[str, List[str]], str]:

        # Load and process image
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        tweet_text = self.tweet_texts[idx]
        
        # Replace [TWEET_TEXT] placeholder with actual tweet text (truncated)
        tweet_text = tweet_text[:200]  # Truncate tweet text
        current_prompt = self.prompt.replace('[TWEET_TEXT]', tweet_text)
        
        # Create message for the image
        message = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": current_prompt},
            ]
        }
        
        prompt = self.processor.apply_chat_template([message], add_generation_prompt=True)
        
        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return inputs, label, self.images[idx]