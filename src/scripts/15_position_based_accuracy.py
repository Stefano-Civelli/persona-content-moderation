import json
import sys
import argparse
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

def load_json(file_path):
    """Load JSON data from file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['results'] if 'results' in data else data

def compute_classification_metrics(y_true, y_pred):
    """
    Compute a dictionary of classification metrics for binary classification.
    Assumes the positive label is `True`.
    """
    if not y_true or not y_pred:
        return None

    metrics = {}

    # Ensure a 2x2 confusion matrix by specifying labels
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()

    metrics['support'] = len(y_true)
    metrics['true_positives'] = tp
    metrics['false_positives'] = fp
    metrics['true_negatives'] = tn
    metrics['false_negatives'] = fn
    metrics['true_hate_samples'] = tp + fn

    # Standard metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, pos_label=True, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, pos_label=True, zero_division=0)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Rate of predicting the positive class (formerly "detection rate")
    if y_pred:
        metrics['positive_prediction_rate'] = sum(1 for pred in y_pred if pred) / len(y_pred)
    else:
        metrics['positive_prediction_rate'] = 0.0

    return metrics

def find_hate_speech_key(results):
    """Find the hate speech label key in the data"""
    hate_keys = ['is_hate', 'is_hate_speech', 'hate_speech', 'hate']
    
    all_label_keys = set()
    for item in results:
        if 'predicted_labels' in item:
            all_label_keys.update(item['predicted_labels'].keys())
    
    # Look for exact matches first
    for key in hate_keys:
        if key in all_label_keys:
            return key
    
    # Look for partial matches
    for key in all_label_keys:
        if any(hate_key in key.lower() for hate_key in hate_keys):
            return key
    
    return None

def print_classification_report(results):
    """Print a comprehensive classification report for the hate speech label."""
    # Find the hate speech label key
    hate_key = find_hate_speech_key(results)
    if not hate_key:
        print("Error: No hate speech label found in the data.")
        all_keys = set()
        for item in results:
            if 'predicted_labels' in item:
                all_keys.update(item['predicted_labels'].keys())
        if all_keys:
            print("Available label keys:", all_keys)
        return

    print(f"Classification Report for '{hate_key}' by Persona Position")
    print("=" * 70)

    # Group by persona_pos
    pos_groups = defaultdict(list)
    for item in results:
        pos = item.get('persona_pos', 'unknown')
        pos_groups[pos].append(item)

    def print_metrics_block(title, items):
        """Helper to compute and print metrics for a given set of items."""
        print(f"\n{title} (n={len(items)})")
        print("-" * 50)

        y_true = []
        y_pred = []
        for item in items:
            if (hate_key in item.get('true_labels', {}) and
                hate_key in item.get('predicted_labels', {})):
                y_true.append(bool(item['true_labels'][hate_key]))
                y_pred.append(bool(item['predicted_labels'][hate_key]))

        if not y_true:
            print("  No data available for this group.")
            return

        metrics = compute_classification_metrics(y_true, y_pred)
        if not metrics:
            print("  Could not compute metrics.")
            return
        
        print(f"  Support: {metrics['support']} samples")
        print(f"    (True Positive Class: {metrics['true_hate_samples']}, True Negative Class: {metrics['support'] - metrics['true_hate_samples']})")
        print(f"  Confusion Matrix:")
        print(f"    TP: {metrics['true_positives']:<5}  FP: {metrics['false_positives']:<5}")
        print(f"    FN: {metrics['false_negatives']:<5}  TN: {metrics['true_negatives']:<5}")
        print("-" * 50)
        print(f"  Accuracy:                 {metrics['accuracy']:.4f}")
        print(f"  Precision (for 'True'):   {metrics['precision']:.4f}")
        print(f"  Recall (for 'True'):      {metrics['recall']:.4f}")
        print(f"  Macro F1-Score:           {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1-Score:        {metrics['weighted_f1']:.4f}")
        print(f"  Positive Prediction Rate: {metrics['positive_prediction_rate']:.4f}  ({metrics['true_positives'] + metrics['false_positives']}/{metrics['support']})")

    # Compute and print metrics for each position
    for pos, items in sorted(pos_groups.items()):
        print_metrics_block(f"Persona Position: {pos}", items)

    # Compute and print overall metrics
    print(f"\n{'='*70}")
    print_metrics_block("Overall Metrics (All Positions)", results)
    print(f"{'='*70}")

def extract_model_name(file_path):
    """Extract model name from file path"""
    parts = file_path.replace('\\', '/').split('/')
    for part in reversed(parts):
        if any(keyword in part.lower() for keyword in ['llama', 'qwen', 'idefics', 'instruct', 'chat']):
            return part
    
    if len(parts) >= 3:
        return parts[-3]
    
    return parts[-1] if parts else file_path

def main():
    parser = argparse.ArgumentParser(description='Analyze classification results')
    parser.add_argument('json_files', nargs='*', 
                       default=[
                           "data/results/img_classification/Idefics3-8B-Llama3/20250813_162245.json/final_results.json", # Facebook
                           "data/results/text_classification/Llama-3.1-8B-Instruct/20250711_112214/final_results.json", # yoder 4-corner
                           #"data/results/text_classification/Llama-3.1-8B-Instruct/20250713_013923/final_results.json", # subdata l-r
                           "data/results/text_classification/Llama-3.1-70B-Instruct/20250724_140425/final_results.json", # yoder 4-corner
                           #"data/results/text_classification/Llama-3.1-70B-Instruct/20250717_084101/final_results.json", # subdata l-r
                           "data/results/text_classification/Qwen2.5-32B-Instruct/20250717_215350/final_results.json", # yoder 4-corner
                           #"data/results/text_classification/Qwen2.5-32B-Instruct/20250713_103728/final_results.json", # subdata l-r
                           "data/results/img_classification/Qwen2.5-VL-7B-Instruct/20250712_012750.json/final_results.json", # Facebook
                           "data/results/img_classification/Qwen2.5-VL-32B-Instruct/20250712_030201/final_results.json", # Facebook
                       ],
                       help='Path(s) to JSON results file(s)')
    
    args = parser.parse_args()
    
    # Process each file
    for i, json_file in enumerate(args.json_files):
        try:
            results = load_json(json_file)
            model_name = extract_model_name(json_file)
            
            print(f"\n{'#' * 80}")
            print(f"MODEL: {model_name}")
            print(f"FILE: {json_file}")
            print(f"{'#' * 80}")
            
            print_classification_report(results)
                
            if i < len(args.json_files) - 1:
                print("\n" + "="*80 + "\n")
                    
        except FileNotFoundError:
            print(f"\n{'#' * 80}")
            print(f"ERROR: File not found - {json_file}")
            print(f"{'#' * 80}")
        except Exception as e:
            print(f"\n{'#' * 80}")
            print(f"ERROR processing {json_file}: {str(e)}")
            print(f"{'#' * 80}")

if __name__ == "__main__":
    main()