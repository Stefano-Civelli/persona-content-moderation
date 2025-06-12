from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch

@dataclass
class ModelConfig:
    """Configuration for model initialization."""

    model_id: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16
    additional_params: Dict[str, Any] = field(default_factory=dict)

class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, config: ModelConfig):
        self.config = config
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