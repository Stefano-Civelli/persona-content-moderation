from src.models.base import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import torch
from typing import List, Any
from vllm.model_executor.utils import set_random_seed
from src.datasets.yoder_text_dataset import IdentityContentClassification
import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VLLMModel(BaseModel):
    """VLLM model implementation for text classification."""

    def setup_model(self) -> None:
        """Initialize VLLM with guided decoding."""
        set_random_seed(self.additional_params.get("seed", 22))
        json_schema_class = self.additional_params.get("json_schema_class", None)
        HF_CACHE = '/scratch/user/uqscivel/HF-CACHE'

        base_llm_kwargs = {
            "tokenizer_mode": "mistral" if "mistral" in self.model_id else "auto",
            "trust_remote_code": True,
            "enforce_eager": False,
            "dtype": "auto",
            "gpu_memory_utilization": 0.95,
            "tensor_parallel_size": 1,
            "enable_prefix_caching": True,
            "disable_log_stats": True,
            "download_dir": HF_CACHE,
        }

        final_llm_kwargs = {
            "model": self.model_id,
            **base_llm_kwargs,
            **self.additional_params
        }

        final_llm_kwargs.pop("seed", None)
        final_llm_kwargs.pop("max_tokens", None)
        final_llm_kwargs.pop("json_schema_class", None)

        logger.info("=" * 70)
        logger.info("VLLM Parameters:")
        logger.info(final_llm_kwargs)
        logger.info("=" * 70 + "\n")

        self.llm = LLM(**final_llm_kwargs)

        # Set up guided decoding for structured output
        self.json_schema = json_schema_class.model_json_schema()
        self.guided_decoding_params = GuidedDecodingParams(json=self.json_schema)
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
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
