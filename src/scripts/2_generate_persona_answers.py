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
        formatted.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    return formatted


def run_inference(llm, tokenizer, prompts: List[str]) -> List[str]:
    # Apply chat template
    formatted_prompts = apply_chat_template(tokenizer, prompts)

    # Set up constrained generation
    choices = ["Disagree", "Agree", "Strongly disagree", "Strongly agree"]
    guided_params = GuidedDecodingParams(choice=choices)
    sampling_params = SamplingParams(guided_decoding=guided_params, temperature=0.0)

    # Generate
    outputs = llm.generate(
        formatted_prompts, sampling_params=sampling_params, use_tqdm=True
    )

    # Extract responses
    return [output.outputs[0].text.strip() for output in outputs]


# ======================================= MAIN =======================================
def main():
    # Configuration
    TORCH_CUDA_ARCH_LIST = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if not TORCH_CUDA_ARCH_LIST:
        raise ValueError("\nTORCH_CUDA_ARCH_LIST environment variable not set")
    print(f"\nUsing TORCH_CUDA_ARCH_LIST: {TORCH_CUDA_ARCH_LIST}")

    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not set")
    login(token=HF_TOKEN)

    HF_CACHE = "/home/pietro/HF-CACHE"

    # Settings
    MODELS = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "HuggingFaceH4/zephyr-7b-beta",
    ]
    SELECTED_MODEL = 0
    MODEL = MODELS[SELECTED_MODEL]

    DATA_PATH = "../../data/processed/base_political_compass_prompts.pqt"
    PERSONAS_PER_BATCH = 10000  # How many personas to process at once
    STATEMENTS_PER_PERSONA = 62  # Fixed based on dataset structure
    VERSION = ""
    # VERSION="_no_chat_template"

    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(DATA_PATH)
    # df = df.head(1240000)  # Remove for full dataset
    print(f"Total prompts: {len(df)}")
    print(f"Total personas: {len(df) // STATEMENTS_PER_PERSONA}")

    # Initialize model and tokenizer
    print(f"\nLoading model: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    llm = LLM(
        model=MODEL,
        tokenizer_mode="mistral" if "mistral" in MODEL else "auto",
        trust_remote_code=True,
        max_model_len=400,  # INPUT + OUTPUT LENGTH
        gpu_memory_utilization=0.95,
        enforce_eager=True,  # If True forces PyTorch to use eager execution mode instead of CUDA graphs, with False will potentially improving performance
        dtype="auto",
        max_num_seqs=(
            250 if "Qwen" in MODEL else 110
        ),  # Maximum concurrency for 400 tokens per request: 126.72x for Llama, 294.28x for Qwen, 149.52x for Mistral, for 143.32x Zephyr
        enable_prefix_caching=True,
        seed=22,
        disable_log_stats=True,
        download_dir=HF_CACHE,
        task="generate",
    )

    # Create output directory
    output_dir = Path(f"../../results/{MODEL.split('/')[-1]}/base/batches{VERSION}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process in batches
    total_personas = len(df) // STATEMENTS_PER_PERSONA

    for i in tqdm(
        range(0, total_personas, PERSONAS_PER_BATCH), desc="Processing batches"
    ):
        # Get batch of personas
        start_idx = i * STATEMENTS_PER_PERSONA
        end_idx = min((i + PERSONAS_PER_BATCH) * STATEMENTS_PER_PERSONA, len(df))
        batch_df = df[start_idx:end_idx].copy()
        batch_prompts = batch_df["prompt"].tolist()

        # Run inference
        try:
            responses = run_inference(llm, tokenizer, batch_prompts)
            batch_df["response"] = responses

            # Save only this batch
            save_path = (
                output_dir
                / f"batch_personas_{i}_to_{min(i + PERSONAS_PER_BATCH, total_personas)}.pqt"
            )
            batch_df.to_parquet(save_path)
            print("\nSaved batch")

        except Exception as e:
            print(f"Error in batch {i}: {e}")
            torch.cuda.empty_cache()
            continue

    print("Done!")

    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
