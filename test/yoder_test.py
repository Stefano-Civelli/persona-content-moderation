import json
import pandas as pd

data_path = "data/raw/yoder_data/sampled/identity_hate_corpora_with_id.jsonl"

with open(data_path, "r") as f:
    data = [json.loads(line) for line in f]

data_df = pd.DataFrame(data)

size_before_deduplication = data_df.shape[0]

# eliminate duplicated rows using fold and text columns
data_df = data_df.drop_duplicates(
    subset=["fold", "text"], keep="first"
)
print(
    f"Removed {size_before_deduplication - data_df.shape[0]} duplicate rows."
)

if "test":
    data_df = data_df[data_df["fold"] == "test"]


data_df = data_df[
    data_df["target_groups"].apply(
        lambda x: len(x) <= 1
    )
]

# save data_df to a new jsonl file
output_path = "./tallaritaitaita.jsonl"
data_df.to_json(output_path, orient="records", lines=True)