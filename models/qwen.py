import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, List
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# PROBABLY DEPRECATED

# This is the utility function from Qwen's codebase that we'll implement directly
def process_vision_info(messages):
    """Extract image and video information from messages."""
    image_list = []
    video_list = []
    for message in messages:
        if isinstance(message.get("content"), list):
            for item in message["content"]:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        image_list.append(item["image"])
                    elif item.get("type") == "video":
                        video_list.append(item["video"])
    return image_list, video_list


class MemeClassifier:
    def __init__(
        self, model_id: str, resolution_factor: int = 4, max_new_tokens: int = 50
    ):
        self.model_id = model_id
        self.resolution_factor = resolution_factor
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_model()

    def setup_model(self) -> None:
        """Initialize the model and processor"""
        # Initialize processor with optional pixel constraints based on resolution_factor
        min_pixels = (256 * 28 * 28) // self.resolution_factor
        max_pixels = (1280 * 28 * 28) // self.resolution_factor

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, min_pixels=min_pixels, max_pixels=max_pixels
        )

        # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        # padding_token_id = tokenizer.pad_token_id
        # padding_side = tokenizer.padding_side
        # print(f'padding_token_id: {padding_token_id}')
        # print(f'padding_side: {padding_side}')
        # assert False

        # Try loading model with flash attention first
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            logging.info("Successfully loaded model with flash attention 2")
        except Exception as e:
            logging.warning(
                f"Failed to load model with flash attention 2: {e}. Falling back to default attention implementation"
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16, device_map="auto"
            )

    def process_batch(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """Process a batch of inputs"""
        # Move batch to device
        batch_inputs = {k: v.to(self.device) for k, v in batch.items()}

        # Generate predictions
        with torch.no_grad():
            generated_ids = self.model.generate(
                **batch_inputs, max_new_tokens=self.max_new_tokens
            )

        # Trim the input tokens from the generated ids
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(batch_inputs["input_ids"], generated_ids)
        ]

        # Decode predictions
        predictions = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return predictions

    def get_processor(self):
        """Return the processor for dataset initialization"""
        return self.processor
