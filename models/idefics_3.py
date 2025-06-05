import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
class MemeClassifier:
    def __init__(self, model_id: str, resolution_factor: int = 4, max_new_tokens: int = 50):
        self.model_id = model_id
        self.resolution_factor = resolution_factor
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_model()
    
    def setup_model(self) -> None:
        """Initialize the model and processor"""

        # This is a Idefics3Processor because 
        # Idefics3Processor offers all the functionalities of Idefics3ImageProcessor.
        # So no need to instantiate both
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            size={"longest_edge": self.resolution_factor * 364},
        )

        # tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3")
        # padding_token_id = tokenizer.pad_token_id
        # padding_side = tokenizer.padding_side
        # print(f'padding_token_id: {padding_token_id}')
        # print(f'padding_side: {padding_side}')
        # assert False
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
    def process_batch(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """Process a batch of inputs"""
        # Move batch to device
        batch_inputs = {k: v.to(self.device) for k, v in batch.items()}
        
        # Generate predictions
        with torch.no_grad():
            generated_ids = self.model.generate(
                **batch_inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode predictions
        predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        return [pred.split("Assistant:")[1].strip() for pred in predictions]

    def get_processor(self):
        """Return the processor for dataset initialization"""
        return self.processor