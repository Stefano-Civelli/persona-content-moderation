import pandas as pd

path = "/scratch/user/uqscivel/extension-llm-political-personas/data/processed/base_political_compass_prompts.pqt"

df = pd.read_parquet(path)

persona_ids = [69740, 160131, 118899]

for persona_id in persona_ids:
    persona = df[df["persona_id"] == int(persona_id)].iloc[0]
    print(persona["cleaned_persona"])
    print(persona["persona_id"])
    print()
    print("--------------------------")