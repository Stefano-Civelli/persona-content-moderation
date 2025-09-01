import json
import csv
import pandas as pd
import sys
import os

def update_labels(json_file_path, csv_file_paths, output_file_path):
    # Read the CSV file and create a mapping from image_name to target
    target_mapping = {}
    try:
        df = pd.DataFrame()
        
        for labels_path in csv_file_paths:
            if labels_path:
                df = pd.concat([df, pd.read_csv(labels_path)], ignore_index=True)
        for _, row in df.iterrows():
            # Extract the image name without extension (assuming .png extension)
            image_name = row['image_name'].replace('.png', '').replace('.jpg', '')
            target_mapping[image_name] = row['target']
        print(f"Loaded {len(target_mapping)} target mappings from CSV")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    trump_count = len(df[df['target'] == 'Trump'])
    hillary_count = len(df[df['target'] == 'Hillary'])
    print(f"Offensive items - Trump: {trump_count}, Hillary: {hillary_count}")
    
    # Read the JSON file
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded JSON file with {len(data['results'])} results")
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return
        
    
    # Update the true_labels
    updated_count = 0
    not_found_count = 0
    
    for result in data['results']:
        item_id = result['item_id']
        
        # Set is_hate_speech to true for all items
        result['true_labels']['is_hate_speech'] = True
        result['true_labels'].pop('attack_method', None)
        
        # Update target_group based on CSV mapping
        if item_id in target_mapping:
            result['true_labels']['target_group'] = target_mapping[item_id]
            updated_count += 1
        else:
            # Keep existing target_group if item_id not found in CSV
            print(f"Warning: item_id '{item_id}' not found in CSV, keeping existing target_group")
            not_found_count += 1
    
    # Write the updated JSON file
    try:
        with open(output_file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully updated JSON file saved to: {output_file_path}")
        print(f"Updated {updated_count} items, {not_found_count} items not found in CSV")
    except Exception as e:
        print(f"Error writing updated JSON file: {e}")

def main():
    json_file_path = "data/results/img_classification/Qwen2.5-VL-32B-Instruct/20250726_172844.json/final_results.json"
    csv_file_paths = [
            "data/interim/MultiOFF_Dataset/Split Dataset/Training_meme_dataset.csv",
            "data/interim/MultiOFF_Dataset/Split Dataset/Validation_meme_dataset.csv",
            "data/interim/MultiOFF_Dataset/Split Dataset/Testing_meme_dataset.csv"
        ]
    output_file_path = "data/results/img_classification/Qwen2.5-VL-32B-Instruct/20250726_172844_fix.json/final_results.json"
    # make output directory if doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    update_labels(json_file_path, csv_file_paths, output_file_path)

if __name__ == "__main__":
    main()