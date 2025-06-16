from typing import Dict, List, Any

import torch
from src.models.base import BaseModel
from transformers import AutoProcessor, AutoModelForVision2Seq

class Idefics3Model(BaseModel):
    """Idefics3 model implementation."""

    def setup_model(self) -> None:
        """Initialize Idefics3 model and processor."""
        
        resolution_factor = self.additional_params.get("resolution_factor", 4)

        # This is a Idefics3Processor because 
        # Idefics3Processor offers all the functionalities of Idefics3ImageProcessor.
        # So no need to instantiate both
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            size={"longest_edge": resolution_factor * 364},
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map="auto",
        )

    def process_batch(self, batch: Dict[str, torch.Tensor]) -> List[str]:
        """Process a batch of inputs."""
        batch_inputs = {k: v.to(self.device) for k, v in batch.items()}

        max_new_tokens = self.additional_params.get("max_new_tokens", 50)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        predictions = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        return [pred.split("Assistant:")[-1].strip() for pred in predictions]

    def prepare_inputs(self, item: Any) -> Any:
        """Prepare inputs - handled by dataset in this case."""
        return item

    def get_processor(self):
        """Return the processor for dataset initialization."""
        return self.processor