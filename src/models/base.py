from abc import ABC, abstractmethod
from typing import Any, List

import torch

class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
        temperature: float = 0.0,
        top_p: float = 1.0,

        **additional_params: Any
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.temperature = temperature
        self.top_p = top_p
        self.additional_params = additional_params
        self.setup_model()

    @abstractmethod
    def setup_model(self) -> None:
        """Initialize model and processor."""
        pass

    @abstractmethod
    def process_batch(self, batch: Any) -> List[str]:
        """Process a batch and return predictions."""
        pass

    @abstractmethod
    def prepare_inputs(self, item: Any) -> Any:
        """Prepare inputs for the model."""
        pass