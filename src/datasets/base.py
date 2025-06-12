from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base class for all datasets."""

    def __init__(
        self, 
        data_path: str,
        max_samples: Optional[int] = None,
        seed: int = 42,
        **additional_params: Any
    ):
        self.data_path = data_path
        self.max_samples = max_samples
        self.seed = seed
        self.additional_params = additional_params
        self.items = []
        self.load_dataset()

    @abstractmethod
    def load_dataset(self) -> None:
        """Load dataset items into self.items."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any], str]:
        """Return (input, labels, item_id)."""
        pass

    def __len__(self) -> int:
        return len(self.items)


class PredictionParser(ABC):
    """Abstract base class for parsing model predictions."""

    @abstractmethod
    def parse(self, prediction: str) -> Dict[str, Any]:
        """Parse raw prediction string into structured labels."""
        pass


class LabelConverter(ABC):
    """Abstract base class for converting dataset labels."""

    @abstractmethod
    def convert(self, raw_labels: Any) -> Dict[str, Any]:
        """Convert raw dataset labels to standardized format."""
        pass
