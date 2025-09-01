PROMPT_TEMPLATE_IDENTIFY_TARGET = '''Determine if the statement is **against** Hillary Clinton or Donald Trump.

**Rules:**
- Say **Hillary** if the statement is against Hillary Clinton
- Say **Trump** if the statement is against Donald Trump  
- Say **Neither** if the statement is not against either person (even if they are mentioned)

**Statement:**
[STATEMENT]

**Answer with exactly one word: Hillary, Trump, or Neither**'''

# -----------------------------------------------------------------------------------------------------------------------------------------


import os
import sys
import pandas as pd
import argparse
from tqdm import tqdm
import openai
import random
import numpy as np
import json
from datetime import datetime
import hashlib
from dotenv import load_dotenv
load_dotenv()
# -------------------- 1. PATH SETUP & IMPORTS --------------------
PROJ_ROOT = os.path.abspath(os.path.join(os.pardir, os.pardir))
PROJ_ROOT_2 = os.path.abspath(os.path.join(os.pardir))
sys.path.append(PROJ_ROOT)
sys.path.append(PROJ_ROOT_2)
print(f'Project root: {PROJ_ROOT}')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def generate_response(prompt, model_name, seed=42, api_key=None, **kwargs):
    """Generate a single response using the OpenAI API with model-specific parameters"""
    
    # Set up OpenAI client
    client = openai.OpenAI(api_key=api_key)
    set_seed(seed)
    
    try:
        # Create the API request with model-specific parameters
        request_params = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            **kwargs  # Include all model-specific parameters from config
        }
        
        # Create the ChatGPT API request
        response = client.chat.completions.create(**request_params)
        
        # Extract the response text
        model_response = response.choices[0].message.content.strip()
        
        # Extract token usage
        token_usage = response.usage.total_tokens
        
        return model_response, token_usage
        
    except Exception as e:
        print(f"Error processing prompt: {e}")
        return f"ERROR: {str(e)}", 0
    

def main():
    parser = argparse.ArgumentParser(description="Generate responses to prompts using ChatGPT API with token tracking")
    
    parser.add_argument(
        "--input_file", 
        type=str,
        default="data/raw/MultiOFF_Dataset/Split Dataset/Testing_meme_dataset.csv",
        help="Path to the CSV file containing prompts in a 'prompt' column"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4.1-mini",
        help="OpenAI model to use (e.g., gpt-4.5-preview, gpt-3.5-turbo, o3-mini)"
    )
    
    parser.add_argument(
        "--api_key",
        type=str,
        help="OpenAI API key (if not provided, will use OPENAI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--token_limit", 
        type=int, 
        default=10000000, # 10 million tokens
        help="Maximum tokens to use before stopping"
    )

    args = parser.parse_args()
    
    # Set the random seed for prompt selection
    set_seed(args.seed)
    model_name_clean = args.model.replace("/", "-").replace(".", "-")

    # Handle API key ---------------------------------------
    if args.api_key is None:
        args.api_key = os.environ.get("OPENAI_API_KEY")
        print(f"Using API key from environment variable: {args.api_key is not None}")
        print(f"KEY: {args.api_key}\n\n")  # Print only the first 4 characters for security
        if args.api_key is None:
            print("Error: No API key provided. Either use --api_key or set the OPENAI_API_KEY environment variable.")
            return
    # Handle API key ---------------------------------------

    # Model Config ---------------------------------------
    model_configs = {
            "gpt-3.5-turbo": {
                "max_tokens": 2048,
                "temperature": 0.0,
                "top_p": 1.0
            },
            "gpt-4-turbo": {
                "max_tokens": 4096,
                "temperature": 0.0,
                "top_p": 1.0
            },
            "o3-mini": {
                "max_tokens": 2048,
                "reasoning_effort": 0.5
            },
            "gpt-4.1-mini": {
                "max_tokens": 512,
                "temperature": 0.0,
                "top_p": 1.0
            },
        }
    
    if args.model not in model_configs:
        print(f"Warning: Model '{args.model}' not found in config. Using default parameters.")
        model_params = {
            "max_tokens": 2048,
            "temperature": 0.0,
            "top_p": 1.0
        }
    else:
        model_params = model_configs[args.model]
        print(f"Using parameters for model '{args.model}': {model_params}")
    # Model Config ---------------------------------------

    
    df = pd.read_csv(args.input_file)
    input_df = df.copy()

    # Reset index
    input_df.reset_index(drop=True, inplace=True)
    # add the "target" column
    input_df['target'] = ' '

    print(f"Loaded {len(input_df)} personas from {args.input_file}")
    
    total_tokens_used = 0
    print(f"Starting new token tracking session (limit: {args.token_limit})")
    
    for idx, row in tqdm(input_df.iterrows(), total=len(input_df)):
        # Check if we've hit the token limit
        if args.token_limit and total_tokens_used >= args.token_limit:
            print(f"\nToken limit reached: {total_tokens_used}/{args.token_limit}")
            break

        # Skip if already cleaned
        if row['target']!=' ':
            continue

        prompt = PROMPT_TEMPLATE_IDENTIFY_TARGET.replace('[STATEMENT]', row['sentence'])
        
        # Generate response
        response, tokens = generate_response(
            prompt=prompt,
            model_name=args.model,
            seed=args.seed,
            api_key=args.api_key,
            **model_params  # Pass model-specific parameters
        )
        
        # Update token count
        total_tokens_used += tokens
        
        # Write the cleaned persona back to the DataFrame
        input_df.at[idx, 'target'] = response

    
    # Save final results ---------------------------------------------------
    output_file = args.input_file.replace('data/raw', f'data/interim')
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    input_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
   

if __name__ == "__main__":
    main()    