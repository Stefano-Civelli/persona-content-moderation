from src.models.base import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import torch
from typing import List, Any
from vllm.model_executor.utils import set_random_seed
from src.datasets.yoder_text_dataset import IdentityContentClassification

class VLLMModel(BaseModel):
    """VLLM model implementation for text classification."""

    def setup_model(self) -> None:
        """Initialize VLLM with guided decoding."""
        set_random_seed(self.additional_params.get("seed", 22))
        # Initialize VLLM
        self.llm = LLM(
            model=self.model_id,
            tokenizer_mode="mistral" if "mistral" in self.model_id else "auto",
            trust_remote_code=True,
            enforce_eager=self.additional_params.get("enforce_eager", False),
            dtype="auto",
            gpu_memory_utilization=0.95,
            tensor_parallel_size=1,
            enable_prefix_caching=True,
            disable_log_stats=True,
            max_model_len=self.additional_params.get("max_model_len", 400),
            max_num_seqs=self.additional_params.get("max_num_seqs", 120)
        )

        # Set up guided decoding for structured output
        self.json_schema = IdentityContentClassification.model_json_schema()
        self.guided_decoding_params = GuidedDecodingParams(json=self.json_schema)

        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=self.additional_params.get("temperature", 0.0),
            top_p=self.additional_params.get("top_p", 1.0),
            max_tokens=self.additional_params.get("max_tokens", 512),
            guided_decoding=self.guided_decoding_params,
        )

    def process_batch(self, prompts: List[str]) -> List[str]:
        """Process a batch of prompts and return predictions."""
        # Generate
        outputs = self.llm.generate(prompts, self.sampling_params)

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
