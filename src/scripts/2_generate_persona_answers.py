import os
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import torch
from huggingface_hub import login
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

def apply_chat_template(tokenizer, prompts: List[str]) -> List[str]:
    """Apply chat template to prompts."""
    formatted = []
    for prompt in prompts:
        messages = [
            {"role": "user", "content": prompt},
        ]
        formatted.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return formatted

def run_inference(llm, tokenizer, prompts: List[str]) -> List[str]:
    """Run inference on prompts with constrained generation."""
    # Apply chat template
    formatted_prompts = apply_chat_template(tokenizer, prompts)
    
    # Set up constrained generation
    choices = ["Disagree", "Agree", "Strongly disagree", "Strongly agree"]
    guided_params = GuidedDecodingParams(choice=choices)
    sampling_params = SamplingParams(guided_decoding=guided_params, temperature=0.0)
    
    # Generate
    outputs = llm.generate(formatted_prompts, sampling_params=sampling_params)
    
    # Extract responses
    return [output.outputs[0].text.strip() for output in outputs]

def main():
    # Configuration
    HF_TOKEN = os.environ.get('HF_TOKEN')
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not set")
    
    login(token=HF_TOKEN)
    
    # Settings
    MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    # MODEL = "Qwen/Qwen2.5-7B-Instruct"
    DATA_PATH = '../../data/processed/political_compass_prompts_200k_personas_noNewline.pqt'
    PERSONAS_PER_BATCH = 100  # How many personas to process at once
    STATEMENTS_PER_PERSONA = 62  # Fixed based on dataset structure
    
    # Load data
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    # df = df.head(2000)  # Remove for full dataset
    print(f"Total prompts: {len(df)}")
    print(f"Total personas: {len(df) // STATEMENTS_PER_PERSONA}")
    
    # Initialize model and tokenizer
    print(f"Loading model: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        model=MODEL,
        max_model_len=400,  # INPUT + OUTPUT LENGTH
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        dtype='auto',
        max_num_seqs=80, 
        enable_prefix_caching=True,
        seed=22,
        disable_log_stats=True
    )
    
    # Create output directory
    output_dir = Path(f"../../data/results/{MODEL.split('/')[-1]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process in batches
    total_personas = len(df) // STATEMENTS_PER_PERSONA
 
    
    for i in tqdm(range(0, total_personas, PERSONAS_PER_BATCH), desc="Processing batches"):
        # Get batch of personas
        start_idx = i * STATEMENTS_PER_PERSONA
        end_idx = min((i + PERSONAS_PER_BATCH) * STATEMENTS_PER_PERSONA, len(df))
        batch_df = df[start_idx:end_idx].copy()
        batch_prompts = batch_df['prompt'].tolist()
        
        # Run inference
        try:
            responses = run_inference(llm, tokenizer, batch_prompts)
            batch_df['response'] = responses
            
            # Save only this batch
            save_path = output_dir / f"batch_personas_{i}_to_{min(i + PERSONAS_PER_BATCH, total_personas)}.pqt"
            batch_df.to_parquet(save_path)
            print("Saved batch")
            

            
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            torch.cuda.empty_cache()
            continue
    
    print("Done!")

    print(f"Results saved in: {output_dir}")

if __name__ == '__main__':
    main()