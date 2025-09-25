from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import Dataset
import pandas as pd
from tqdm.auto import tqdm
import pickle
import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC):
    """Abstract base class for all datasets."""

    def __init__(
        self, 
        data_path: str,
        prompts_file: Optional[str] = None,
        max_samples: Optional[int] = None,
        extreme_pos_personas_path: Optional[str] = None,
        prompt_template: Optional[str] = None,
        seed: int = 42,
        **additional_params: Any
    ):
        self.data_path = data_path
        self.prompts_file = prompts_file
        self.prompts = {}
        self.max_samples = max_samples
        self.extreme_pos_personas_path = extreme_pos_personas_path
        self.prompt_template = prompt_template
        self.seed = seed
        self.additional_params = additional_params
        self.items = []
        self.load_dataset()
        self.load_and_build_prompts()


    def _load_prompts(self) -> None:
        """Load prompts from file."""
        prompts_df = pd.read_parquet(self.prompts_file)
        for _, row in prompts_df.iterrows():
            self.prompts[row["persona_id"]] = (row["prompt"], row["persona_pos"])
        self.persona_ids = list(self.prompts.keys())

    def load_and_build_prompts(self):
        logger.info("=" * 70)
        # Handle the case where no personas are used (baseline)
        if self.extreme_pos_personas_path is None:
            logger.info("No personas path provided. Using baseline setup with default prompt...")
            
            # Create a single default prompt entry
            default_persona_id = "default"
            default_prompt = self.prompt_template
            
            self.prompts[default_persona_id] = (default_prompt, "baseline")
            self.persona_ids = [default_persona_id]
            
            logger.info(f"Created baseline setup with default prompt: {default_prompt}")
            logger.info("=" * 70 + "\n")
            return
        
        # Original persona loading logic
        logger.info("Loading extreme personas and building prompts...")
        with open(self.extreme_pos_personas_path, 'rb') as f:
            extreme_pos_personas = pickle.load(f)
        extreme_personas_df = create_personas_df(extreme_pos_personas)
        logger.info(f"Loaded {len(extreme_personas_df)} extreme personas.")
        logger.info(f"Sample extreme persona: {extreme_personas_df.iloc[0].to_dict()}")
        extreme_personas_list = extreme_personas_df.values.tolist()

        assert len(extreme_personas_df) == 60 or len(extreme_personas_df) == 400

        prompts_df = generate_prompts(self.prompt_template, extreme_personas_list)
        logger.info(f"Generated {len(prompts_df)} prompts.")
        logger.info(f"Sample prompt: {prompts_df.iloc[0].to_dict()}\n")

        for _, row in prompts_df.iterrows():
            self.prompts[row["persona_id"]] = (row["prompt"], row["persona_pos"])
        self.persona_ids = list(self.prompts.keys())
        logger.info("=" * 70 + "\n")
  
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


def create_personas_df(extreme_pos):
    persona_ids = []
    persona_descriptions = []
    persona_pos = []
    
    for quadrant in extreme_pos.keys():
        for persona in extreme_pos[quadrant]:
            persona_ids.append(persona[0])
            persona_descriptions.append(persona[1])
            persona_pos.append(quadrant)
    
    personas_df = pd.DataFrame({
        'persona_id': persona_ids,
        'description': persona_descriptions,
        'pos': persona_pos
    })
    return personas_df



def generate_prompts(prompt_template, personas_list):
    """Generate prompts by replacing [PERSONA] in the template with actual persona descriptions."""
    
    prompts = []

    for persona_id, description, pos in tqdm(personas_list, total=len(personas_list)):
        
        prompt = prompt_template.replace('[PERSONA]', description)
        
        record = {
            'persona_id': persona_id,
            'persona_pos': pos,
            'persona': description,
            'prompt': prompt
        }

        prompts.append(record)

    return pd.DataFrame(prompts)