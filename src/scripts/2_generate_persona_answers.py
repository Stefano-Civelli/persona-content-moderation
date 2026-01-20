# ---------------------------------------------------------
# From Bernardelle et al. political-ideology-shifts-in-LLMs
# availabele at: https://github.com/d-lab/political-ideology-shifts-in-LLMs/blob/main/src/2.LLMsInference/2.LLMsPersonaLeaningInference.py
# --------------------------------------------------------

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import argparse
import socket
import subprocess


# ======================================= MAIN =======================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate responses to political compass prompts using LLMs with VLLM."
    )

    parser.add_argument(
        "--number_of_personas",
        type=int,
        default=None,
        help="Size of the data subset to use for testing.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Number of personas to process in each batch.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use for inference.",
    )

    parser.add_argument(
        "--start_batch",
        type=int,
        default=0,
        help="Starting batch index for processing.",
    )

    args = parser.parse_args()

    BATCH_SIZE = args.batch_size

    START_BATCH = args.start_batch
    NUMBER_OF_SAMPLES = (
        args.number_of_personas * 62 if args.number_of_personas is not None else None
    )

    print()
    print("=" * 70)
    print(f"Using batch size: {BATCH_SIZE} persona")
    print(f"Using model: {args.model}")
    print(f"Starting from batch: {START_BATCH}")
    result = subprocess.run(["hostname"], capture_output=True, text=True, check=True)
    print(f"Running on: {result.stdout.strip()}")
    print("=" * 70)
    print()
    DATA_PATH = "data/cache/elite_political_compass_prompts_2k.pqt"

    PERSONAS_PER_BATCH = BATCH_SIZE  # How many personas to process at once
    STATEMENTS_PER_PERSONA = 62  # Fixed based on dataset structure

    # Load data
    print()
    print("=" * 70)
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)

    if args.number_of_personas is not None:
        df = df.head(NUMBER_OF_SAMPLES)
    print(f"Total prompts: {len(df)}")
    print(f"Total personas: {len(df) // STATEMENTS_PER_PERSONA}")

    # Initialize model and tokenizer
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        #tokenizer_mode="mistral" if "mistral" in args.model else "auto",
        trust_remote_code=True,
        max_model_len=700,  # INPUT + OUTPUT LENGTH
        gpu_memory_utilization=0.9,
        # enforce_eager=True, # If True disables the construction of CUDA graph in Pytorch. This may harm performance but reduces the memory requirement (of maintaining the CUDA graph)
        dtype="auto",
        max_num_seqs=100,
        enable_prefix_caching=True,  # Zephyr does not support prefix caching
        seed=22,
        disable_log_stats=True,
        # if the model fits in a single node with multiple GPUs, but the number of GPUs cannot divide the model size evenly, you can use pipeline parallelism, which splits the model
        # along layers and supports uneven splits. In this case, the tensor parallel size should be 1 and the pipeline parallel size should be the number of GPUs.
        tensor_parallel_size=2 if ("70B" in args.model or "72B" in args.model) else 1,
        pipeline_parallel_size=1,
        # pipeline_parallel_size=3 if '72B' in args.model else 1,
    )

    # Create output directory
    output_dir = Path(f"data/cache/{args.model.split('/')[-1]}/base/batches")
    output_dir.mkdir(parents=True, exist_ok=True)
    print()
    print("=" * 70)
    print(f"Output dir: {output_dir}")
    print("=" * 70)
    print()


    # Set up constrained generation
    choices = ["Disagree", "Agree", "Strongly disagree", "Strongly agree"]
    structured_outputs_params_choice = StructuredOutputsParams(
        choice=choices
    )
    sampling_params = SamplingParams(structured_outputs=structured_outputs_params_choice, temperature=0.0)

    # Process in batches
    total_personas = len(df) // STATEMENTS_PER_PERSONA

    for i in tqdm(
        range(START_BATCH * PERSONAS_PER_BATCH, total_personas, PERSONAS_PER_BATCH),
        desc="Processing batches",
    ):
        print(f"Starting from persona index: {i}")
        # Get batch of personas
        start_idx = i * STATEMENTS_PER_PERSONA
        end_idx = min((i + PERSONAS_PER_BATCH) * STATEMENTS_PER_PERSONA, len(df))
        batch_df = df[start_idx:end_idx].copy()
        batch_prompts = batch_df["prompt"].tolist()
        print(f"\n Example of prompt: {batch_prompts[0]}")

        # Run inference
        try:
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in batch_prompts
            ]
            outputs = llm.generate(
                formatted_prompts, sampling_params=sampling_params, use_tqdm=True
            )
            batch_df["response"] = [
                output.outputs[0].text.strip() for output in outputs
            ]

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
