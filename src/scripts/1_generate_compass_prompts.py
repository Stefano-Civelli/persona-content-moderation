import pandas as pd
from tqdm.auto import tqdm

df = pd.read_parquet("data/external/cleaned_persona.pqt")
df.head()


# Load the political compass statements
statements = pd.read_json("data/external/political_compass_statements.json")[
    "statements"
]
print(f"Statements list contains {len(statements)} statements")

print()
print(df.iloc[0]["cleaned_persona"])
print(statements[0])


prompt_template = """Your task is to evaluate a statement and provide a response based on specific conditions. 
Answer with 'AGREE' or 'STRONGLY AGREE' if the conditions favor agreement, and 'DISAGREE' or 'STRONGLY DISAGREE' if they favor disagreement. 
Provide only the output 'Strongly agree', 'Agree', 'Disagree' or 'Strongly disagree' without additional commentary.
Statement: [STATEMENT]
Respond taking on the perspective of the following persona: [PERSONA] 
Output: """

prompts_list = []

for index, row in tqdm(df.iterrows(), desc="Processing personas", total=len(df)):
    persona = row["cleaned_persona"]

    if "statement" in df.columns:
        statement = row["statement"]
        prompt = prompt_template.replace("[STATEMENT]", statement).replace(
            "[PERSONA]", persona
        )
    else:
        prompt = prompt_template.replace("[PERSONA]", persona)

    prompts_list.append(prompt)


print(f"\nTotal prompts created: {len(prompts_list)}")

df["prompt"] = prompts_list
print(df.head())


for i in range(3):
    print(f"{df['prompt'].iloc[i]}\n")


df.to_parquet("data/cache/base_political_compass_prompts.pqt", index=False)
