import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset(
    "proj-persona/PersonaHub", "elite_persona", split="train", streaming=True
)

N = 200000
# Shuffle with a buffer and take N samples (streaming mode)
dataset = dataset.shuffle(seed=42, buffer_size=10000).take(N)

df = pd.DataFrame(list(dataset))
df.head()


# Load the political compass statements
statements = pd.read_json("data/external/political_compass_statements.json")[
    "statements"
]
print(f"Statements list contains {len(statements)} statements")

print()
print(df.iloc[0]["persona"])
print(statements[0])


prompt_template = """Your task is to evaluate a statement and provide a response based on specific conditions. 
Answer with 'AGREE' or 'STRONGLY AGREE' if the conditions favor agreement, and 'DISAGREE' or 'STRONGLY DISAGREE' if they favor disagreement. 
Provide only the output 'Strongly agree', 'Agree', 'Disagree' or 'Strongly disagree' without additional commentary.
Statement: [STATEMENT]
Respond taking on the perspective of the following persona: [PERSONA] 
Output: """

new_rows = []

# Outer Loop: Iterate through every Persona
for index, row in tqdm(df.iterrows(), desc="Processing combinations", total=len(df)):
    persona = row["persona"]

    # Inner Loop: Iterate through every Statement
    for statement in statements:
        # Create the prompt
        prompt = prompt_template.replace("[STATEMENT]", statement).replace(
            "[PERSONA]", persona
        )

        # Store the data so we keep track of which prompt belongs to which persona/statement
        new_rows.append({"persona": persona, "statement": statement, "prompt": prompt})

# Create the final expanded DataFrame
df_expanded = pd.DataFrame(new_rows)


print(f"\nTotal prompts created: {len(df_expanded)}")


print(df_expanded.head())

for i in range(3):
    print(f"{df_expanded['prompt'].iloc[i]}\n")


df_expanded.to_parquet(
    f"data/cache/elite_political_compass_prompts_{N//1000}k.pqt", index=False
)


# Calculate max token length based on longest components
print("\nCalculating max token length...")

# Find longest components by character length (heuristic)
longest_persona = df.loc[df["persona"].str.len().idxmax()]["persona"]
longest_statement = max(statements, key=len)

longest_prompt = prompt_template.replace("[STATEMENT]", longest_statement).replace(
    "[PERSONA]", longest_persona
)

print(f"Longest persona chars: {len(longest_persona)}")
print(f"Longest statement chars: {len(longest_statement)}")

# Check token length for models
model_ids = {
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
}

for name, model_id in model_ids.items():
    print(f"Loading tokenizer for {name} ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    length = len(tokenizer.encode(longest_prompt))
    print(f"Max possible token length for {name}: {length} tokens")
