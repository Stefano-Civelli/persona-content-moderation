import json
import os
import glob
from pathlib import Path

def merge_json_batches(folder_path, output_path="partial_results.json"):
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Find all JSON files in the folder
    json_files = list(folder_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{folder_path}'")
        return
    
    print(f"Found {len(json_files)} JSON files to merge...")
    
    # sort them by name
    json_files.sort(key=lambda x: int(x.name.split("-")[0].split("_")[-1]))

    # Initialize the merged data structure
    merged_data = {
        "metrics": {},
        "metadata": {
            "task_config": {
            "data_path_yoder": "data/raw/yoder_data/sampled/identity_hate_corpora_with_id.jsonl",
            "data_path_subdata": "data/raw/subdata/political_complete.csv",
            "extreme_pos_path": "data/results/extreme_pos_personas/Llama-3.1-70B-Instruct/extreme_pos_corners_100.pkl",
            "output_path": "data/results/text_classification/Llama-3.1-70B-Instruct/20250713_005408"
            },
            "model_config": {
            "name": "meta-llama/Llama-3.1-70B-Instruct",
            },
            "dataset_name": "yoder",
        },
        "results": []
    }
    
    # Process each JSON file
    for json_file in json_files:
        try:
            print(f"Processing: {json_file.name}")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Merge results array
            if "results" in data and isinstance(data["results"], list):
                merged_data["results"].extend(data["results"])
            
            # Merge metrics (you can customize this logic based on your needs)
            if "metrics" in data and isinstance(data["metrics"], dict):
                merged_data["metrics"].update(data["metrics"])
            
            # Merge metadata (you can customize this logic based on your needs)
            if "metadata" in data and isinstance(data["metadata"], dict):
                merged_data["metadata"].update(data["metadata"])
                
        except json.JSONDecodeError as e:
            print(f"Error reading {json_file.name}: Invalid JSON format - {e}")
            continue
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nMerging complete!")
        print(f"Total results merged: {len(merged_data['results'])}")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving merged file: {e}")

if __name__ == "__main__":
    common_path = "data/results/text_classification/Llama-3.1-70B-Instruct/20250713_005408"
    folder_path = f"{common_path}/batches"
    output_path = f"{common_path}/partial_results.json"
    merge_json_batches(folder_path, output_path)