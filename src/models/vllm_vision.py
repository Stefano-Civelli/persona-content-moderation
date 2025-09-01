from abc import ABC, abstractmethod
from typing import List, Any
import torch
from PIL import Image
from vllm.sampling_params import GuidedDecodingParams
from vllm import LLM, SamplingParams
from vllm.model_executor.utils import set_random_seed
from transformers import AutoTokenizer
import logging
from src.models.base import BaseModel
from src.datasets.facebook_hateful_memes_dataset_vllm import (
    HatefulContentClassification,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VisionVLLMModel(BaseModel, ABC):
    """
    Abstract base class for Vision-Language models using vLLM.
    """

    def get_model_specific_config(self) -> dict:
        """Override this method to provide model-specific configuration."""
        return {}

    def get_stop_token_ids(self) -> List[int]:
        """Override this method to provide model-specific stop tokens."""
        return None

    def setup_model(self) -> None:
        """Initialize the VLLM engine."""
        set_random_seed(self.additional_params.get("seed", 22))
        json_schema_class = self.additional_params.get("json_schema_class", None)
        HF_CACHE = "/scratch/user/uqscivel/HF-CACHE"

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
            "limit_mm_per_prompt": {"image": 1},  # 1 image per prompt
        }

        # Apply model-specific configuration
        base_llm_kwargs.update(self.get_model_specific_config())

        # Additional_params comes last so it can override the defaults.
        final_llm_kwargs = {
            "model": self.model_id,
            **base_llm_kwargs,
            **self.additional_params,
        }

        final_llm_kwargs.pop("seed", None)
        final_llm_kwargs.pop("max_tokens", None)
        final_llm_kwargs.pop("json_schema_class", None)
        final_llm_kwargs.pop("resolution_factor", None)

        logger.info("=" * 70)
        logger.info("VLLM Parameters:")
        logger.info(final_llm_kwargs)
        logger.info("=" * 70 + "\n")

        self.llm = LLM(**final_llm_kwargs)

        self.json_schema = json_schema_class.model_json_schema()
        self.guided_decoding_params = GuidedDecodingParams(json=self.json_schema)

        # Get model-specific stop tokens
        stop_token_ids = self.get_stop_token_ids()

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.additional_params.get("max_tokens", 512),
            guided_decoding=self.guided_decoding_params,
            stop_token_ids=stop_token_ids if stop_token_ids else None,
        )

    @abstractmethod
    def _format_prompt(self, text: str) -> str:
        pass

    def process_batch(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        if len(prompts) != len(images):
            raise ValueError(
                "The number of prompts and images in the batch must be the same."
            )

        # Create the list of inputs in the format vLLM expects
        vllm_inputs = []
        for i in range(len(prompts)):
            formatted_prompt = self._format_prompt(prompts[i])
            vllm_inputs.append(
                {
                    "prompt": formatted_prompt,
                    "multi_modal_data": {"image": images[i]},
                }
            )

        # Generate responses
        outputs = self.llm.generate(vllm_inputs, self.sampling_params)

        predictions = [output.outputs[0].text for output in outputs]
        return predictions

    def prepare_inputs(self, item: Any) -> Any:
        """Not used for this batch-based pipeline."""
        return item

    def cleanup(self):
        """Release resources used by the LLM."""
        if hasattr(self, "llm"):
            del self.llm
            torch.cuda.empty_cache()


# ===================================================================
# Concrete Model Implementations Inspired by vision_language.py from vLLM Docs
# https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language.py
# ===================================================================


class MiniCPMVLLMModel(VisionVLLMModel):
    """
    VLLM implementation for MiniCPM-o-2_6 models.
    """

    def get_model_specific_config(self) -> dict:
        """MiniCPM-specific configuration."""
        return {
            "max_model_len": 4096,
            "max_num_seqs": 2,
        }

    def get_stop_token_ids(self) -> List[int]:
        """Get MiniCPM-specific stop tokens."""
        if not hasattr(self, "_stop_token_ids"):
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            stop_tokens = ["<|im_end|>", "<|endoftext|>"]
            self._stop_token_ids = [
                tokenizer.convert_tokens_to_ids(token) for token in stop_tokens
            ]
        return self._stop_token_ids

    def _format_prompt(self, text: str) -> str:
        """Format prompt using MiniCPM chat template."""
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )

        messages = [
            {
                "role": "user",
                "content": f"(<image>./</image>)\n{text}",
            }
        ]

        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


      
class Idefics3VLLMModel(VisionVLLMModel):
    """
    VLLM implementation for Idefics3 models, with support for custom resolution.
    """

    def get_model_specific_config(self) -> dict:

        resolution_factor = self.additional_params.get("resolution_factor", 4)

        target_longest_edge = resolution_factor * 364
        
        logger.info(
            f"Configuring Idefics3 image resolution with longest_edge: {target_longest_edge}"
        )

        config = {
            "mm_processor_kwargs": {
                "size": {"longest_edge": target_longest_edge},
            }
        }
        
        return config

    def _format_prompt(self, text: str) -> str:
        return f"<|begin_of_text|>User:<image>{text}<end_of_utterance>\nAssistant:"


class QwenVL2_5VLLMModel(VisionVLLMModel):
    """
    VLLM implementation for Qwen2.5-VL models.
    """

    def _format_prompt(self, text: str) -> str:
        # Note the use of <|image_pad|> as the placeholder.
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
