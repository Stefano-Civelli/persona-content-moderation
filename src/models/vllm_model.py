from src.models.base import BaseModel
from vllm import LLM, SamplingParams
from vllm.model_executor.guided_decoding import GuidedDecodingParams
from transformers import AutoTokenizer
import torch
from typing import List, Any
from src.datasets.hate_speech_text_dataset import ContentClassification
from vllm.model_executor.utils import set_random_seed


class VLLMModel(BaseModel):
    """VLLM model implementation for text classification."""

    def setup_model(self) -> None:
        """Initialize VLLM with guided decoding."""
        set_random_seed(self.additional_params.get("seed", 22))

        # Initialize VLLM
        self.llm = LLM(
            model=self.model_id,
            trust_remote_code=True,
            dtype="bfloat16" if torch.cuda.is_available() else "float32",
            gpu_memory_utilization=0.95,
            max_model_len=4096,
            tensor_parallel_size=1,
            enable_prefix_caching=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Set up guided decoding for structured output
        self.json_schema = ContentClassification.model_json_schema()
        self.guided_decoding_params = GuidedDecodingParams(json=self.json_schema)

        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=self.additional_params.get("temperature", 0.0),
            top_p=self.additional_params.get("top_p", 1.0),
            max_tokens=self.additional_params.get("max_tokens", 2048),
            guided_decoding=self.guided_decoding_params,
        )

    def process_batch(self, batch: List[str]) -> List[str]:
        """Process a batch of prompts and return predictions."""
        # Apply chat template
        formatted_prompts = []
        for prompt in batch:
            formatted_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted_prompts.append(formatted_prompt)

        # Generate
        outputs = self.llm.generate(formatted_prompts, self.sampling_params)

        # Extract predictions
        predictions = []
        for output in outputs:
            predictions.append(output.outputs[0].text)

        return predictions

    def prepare_inputs(self, item: Any) -> Any:
        """VLLM handles text directly, no special preparation needed."""
        return item

    def cleanup(self):
        """Release resources used by the LLM."""
        if hasattr(self, "llm"):
            del self.llm
            torch.cuda.empty_cache()
