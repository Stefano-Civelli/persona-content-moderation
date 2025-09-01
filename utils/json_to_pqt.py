import json
import pandas as pd

# Define the input and output file paths
input_json_path = 'experiment_data.json'
output_parquet_path = 'results.pqt'
output_metadata_path = 'run_info.json'

print(f"Loading data from {input_json_path}...")

# 1. Load the entire JSON file into memory
with open(input_json_path, 'r') as f:
    data = json.load(f)

# 2. Separate the row-level data from the run-level data
results_data = data['results']
run_metadata = {
    'metrics': data['metrics'],
    'metadata': data['metadata']
}

print(f"Found {len(results_data)} records in the 'results' list.")

# 3. Flatten the 'results' data into a pandas DataFrame
# pandas.json_normalize is perfect for this kind of nested structure.
# We use a separator to create column names like 'true_labels_is_hate_speech'.
df = pd.json_normalize(results_data, sep='_')

# 4. (Optional but recommended) Clean up column names for clarity
# For example, rename 'true_labels_is_hate_speech' to 'true_is_hate_speech'
df = df.rename(columns={
    'true_labels_is_hate_speech': 'true_is_hate_speech',
    'true_labels_target_group': 'true_target_group',
    'true_labels_attack_method': 'true_attack_method',
    'predicted_labels_is_hate_speech': 'predicted_is_hate_speech',
    'predicted_labels_target_group': 'predicted_target_group',
    'predicted_labels_attack_method': 'predicted_attack_method'
})

print("\nDataFrame created with the following columns:")
print(df.info())

# 5. Save the DataFrame to a Parquet file
# The 'pyarrow' engine is used by default. Compression is on by default ('snappy').
df.to_parquet(output_parquet_path, index=False)
print(f"\nSuccessfully saved tabular data to {output_parquet_path}")

# 6. Save the run-level metadata to a separate JSON file
with open(output_metadata_path, 'w') as f:
    json.dump(run_metadata, f, indent=2)
print(f"Successfully saved run metadata to {output_metadata_path}")