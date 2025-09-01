import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import os

def calculate_accuracies(results, accuracy_method='is_hate_speech'):
    stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))

    # Mapping to normalize category names to a standard, uppercase format
    category_map = {
        "blacks": "BLACK", "black": "BLACK",
        "muslims": "MUSLIM", "muslim": "MUSLIM",
        "lgbtq+": "LGBTQ+", "lgbtq": "LGBTQ+",
        "jews": "JEWS", "jew": "JEWS",
        "asians": "ASIAN", "asian": "ASIAN", "asain": "ASIAN",
        "latinx": "LATINX",
        "women": "WOMEN", "woman": "WOMEN",
        "christians": "CHRISTIAN", "christian": "CHRISTIAN",
        "men": "MEN", "man": "MEN",
        "whites": "WHITE", "white": "WHITE",
        "none": "NONE"
    }

    for item in results:
        true_labels = item.get('true_labels', {})
        pred_labels = item.get('predicted_labels', {})
        persona = item.get('persona_pos')

        # Normalize true category name
        true_category_raw = true_labels.get('target_category', 'none').lower()
        true_category = category_map.get(true_category_raw, true_category_raw.upper())

        # We don't calculate accuracy for items where the true label is 'none'
        if true_category == 'NONE':
            continue

        is_correct = False
        if accuracy_method == 'is_hate_speech':
            # Prediction is correct if 'is_hate_speech' boolean matches
            is_correct = (true_labels.get('is_hate_speech') == pred_labels.get('is_hate_speech'))
        elif accuracy_method == 'exact_match':
            # Prediction is correct only if both labels match
            predicted_category_raw = pred_labels.get('target_category', 'none').lower()
            predicted_category = category_map.get(predicted_category_raw, predicted_category_raw.upper())
            
            is_correct = (true_labels.get('is_hate_speech') == pred_labels.get('is_hate_speech') and
                          true_category == predicted_category)
        else:
            raise ValueError("accuracy_method must be 'is_hate_speech' or 'exact_match'")
        
        stats[persona][true_category]['total'] += 1
        if is_correct:
            stats[persona][true_category]['correct'] += 1

    accuracies = defaultdict(dict)
    for persona, categories in stats.items():
        for category, counts in categories.items():
            if counts['total'] > 0:
                acc = (counts['correct'] / counts['total']) * 100
                accuracies[persona][category] = acc
    
    return accuracies

def generate_plot(file_path, accuracy_method):
    """
    Loads data from the specified file, calculates accuracies, and generates a heatmap plot.

    Args:
        file_path (str): Path to the JSON results file.
        accuracy_method (str): Method for accuracy calculation ('is_hate_speech' or 'exact_match').
    """
    # 1. Load and parse the JSON file
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        results = data.get('results', [])
        if not results:
            print("Error: 'results' key not found or is empty in the JSON file.")
            return
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return

    # 2. Calculate accuracies
    accuracies = calculate_accuracies(results, accuracy_method)
    if not accuracies:
        print("No data to plot after processing.")
        return

    # 3. Create a pandas DataFrame for plotting
    df = pd.DataFrame.from_dict(accuracies, orient='index')
    
    # 4. Define column and row order for consistency
    #row_order = ['bottom_left', 'top_left', 'bottom_right', 'top_right']
    row_order = ['leftmost', 'rightmost']
    
    # Desired column order (protected groups, then majority groups)
    col_order_protected = ['BLACK', 'MUSLIMS/ARABIC', 'LGBTQ+', 'JEWS', 'ASIAN', 'LATINO/HISPANIC', 'WOMEN']
    col_order_majority = ['CHRISTIAN', 'MEN', 'WHITE']
    
    present_cols = df.columns.tolist()
    ordered_cols = [col for col in col_order_protected if col in present_cols] + \
                   [col for col in col_order_majority if col in present_cols]
    
    other_cols = [col for col in present_cols if col not in ordered_cols]
    final_col_order = ordered_cols + sorted(other_cols)
    
    # Reorder the DataFrame rows and columns and fill missing data with 0
    df = df.reindex(index=row_order, columns=final_col_order).fillna(0)
    
    # 5. Generate the heatmap
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(15, 6))
    
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=.5,
        linecolor='white',
        ax=ax,
        cbar_kws={'label': 'Accuracy (%)'}
    )
    
    # 6. Customize the plot
    title = f"Harmfulness Detection Accuracy by Persona and Target Group\n(Method: {accuracy_method.replace('_', ' ').title()})"
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel("Persona Position", fontsize=12)
    ax.set_xlabel(None)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center', fontsize=11)
    
    # Add the vertical separator line
    num_protected_groups = len([col for col in col_order_protected if col in final_col_order])
    if 0 < num_protected_groups < len(final_col_order):
        ax.axvline(x=num_protected_groups, color='white', linestyle='--', linewidth=3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 7. Save the plot
    base_name = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path)
    output_filename = f"accuracy_heatmap_{accuracy_method}_{os.path.splitext(base_name)[0]}.png"
    output_path = os.path.join(dir_name, output_filename)
    
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {output_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default="data/results/text_classification/Qwen2.5-32B-Instruct/20250722_125943/final_results.json"
    )
    parser.add_argument(
        "--accuracy_method",
        type=str,
        default="exact_match",
        choices=["is_hate_speech", "exact_match"],
        help=(
            "The method for calculating accuracy:\n"
            "  'is_hate_speech': Correct if 'is_hate_speech' label matches. (default)\n"
            "  'exact_match': Correct if 'is_hate_speech' AND 'target_category' labels both match."
        )
    )
    
    args = parser.parse_args()
    
    generate_plot(args.file_path, args.accuracy_method)

if __name__ == "__main__":
    main()