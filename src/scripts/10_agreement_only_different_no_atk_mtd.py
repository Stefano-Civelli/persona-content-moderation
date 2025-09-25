import argparse
import json
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import mannwhitneyu
import os
from utils.util import (
    load_config,
    get_model_config,
)
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Manager

DATASET_NAME = None

positions_text_labels = {
    "top_left": "Top-L",
    "top_right": "Top-R",
    "bottom_left": "Bot-L",
    "bottom_right": "Bot-R",
    "leftmost": "Left",
    "rightmost": "Right",
}


def gwets_ac1(y1, y2):
    """
    Compute Gwet's AC1 agreement coefficient.
    
    Args:
        y1, y2: Arrays of ratings from two raters
        
    Returns:
        AC1 coefficient (float)
    """
    y1 = np.array(y1)
    y2 = np.array(y2)
    
    if len(y1) != len(y2):
        raise ValueError("Arrays must have the same length")
    
    n = len(y1)
    if n == 0:
        return np.nan
    
    # Get unique categories
    categories = np.unique(np.concatenate([y1, y2]))
    q = len(categories)
    
    if q == 1:
        return 1.0  # Perfect agreement when only one category
    
    # Create category mapping
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    
    # Convert to indices
    y1_idx = np.array([cat_to_idx[cat] for cat in y1])
    y2_idx = np.array([cat_to_idx[cat] for cat in y2])
    
    # Observed agreement
    po = np.mean(y1_idx == y2_idx)
    
    # Expected agreement under uniform distribution (Gwet's assumption)
    pe_gwet = 1.0 / q
    
    # AC1 coefficient
    if pe_gwet == 1.0:
        return 1.0 if po == 1.0 else 0.0
    
    ac1 = (po - pe_gwet) / (1 - pe_gwet)
    return ac1


def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    
    # Calculate means
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Calculate pooled standard deviation
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    
    # Avoid division by zero
    if pooled_std == 0:
        return 0.0
    
    # Calculate Cohen's d
    d = (mean_x - mean_y) / pooled_std
    return d


def interpret_cohens_d(d):

    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def perform_mann_whitney_test(group1, group2, label_name, pos1, pos2):
    try:
        # Perform Mann-Whitney U test
        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        
        # Calculate Cohen's d
        effect_size = cohens_d(group1, group2)
        effect_interpretation = interpret_cohens_d(effect_size)
        
        return {
            'label': label_name,
            'position_1': pos1,
            'position_2': pos2,
            'n1': len(group1),
            'n2': len(group2),
            'mean_1': np.mean(group1),
            'mean_2': np.mean(group2),
            'std_1': np.std(group1, ddof=1),
            'std_2': np.std(group2, ddof=1),
            'statistic': statistic,
            'p_value': p_value,
            'cohens_d': effect_size,
            'effect_interpretation': effect_interpretation,
            'significant': p_value < 0.05
        }
    except Exception as e:
        print(f"Error in Mann-Whitney test for {label_name} between {pos1} and {pos2}: {str(e)}")
        return None


def identify_disagreement_items(df, positions):
    """
    Identify items where personas disagree on 'is_hate_speech' predictions.
    
    Args:
        df: DataFrame with predictions
        positions: List of position names
        
    Returns:
        set: Item IDs where there is disagreement on 'is_hate_speech'
    """
    disagreement_items = set()
    
    # Group by item_id to check predictions across all personas for each item
    for item_id, item_group in df.groupby('item_id'):
        hate_speech_predictions = item_group['is_hate_speech'].unique()
        
        # If there's more than one unique prediction, there's disagreement
        if len(hate_speech_predictions) > 1:
            disagreement_items.add(item_id)
    
    print(f"\nDisagreement Analysis:")
    print(f"Total items: {df['item_id'].nunique()}")
    print(f"Items with disagreement on 'is_hate_speech': {len(disagreement_items)}")
    print(f"Items with unanimous agreement: {df['item_id'].nunique() - len(disagreement_items)}")
    print(f"Percentage of items with disagreement: {len(disagreement_items) / df['item_id'].nunique() * 100:.2f}%")
    
    return disagreement_items


def save_comprehensive_report(kappa_df, ac1_df, raw_agreement_df, descriptive_stats, test_results, positions, output_path, agreement_summary, disagreement_stats):
   
    with open(output_path, 'w') as f:
        f.write("COMPREHENSIVE STATISTICAL ANALYSIS REPORT\n")
        f.write("(Agreement computed only on items with disagreement in 'is_hate_speech' predictions)\n")
        f.write("=" * 80 + "\n\n")
        
        # 0a. Disagreement Statistics
        f.write("0a. DISAGREEMENT FILTERING STATISTICS\n")
        f.write("-" * 40 + "\n\n")
        
        f.write(f"Total items in dataset: {disagreement_stats['total_items']}\n")
        f.write(f"Items with disagreement on 'is_hate_speech': {disagreement_stats['disagreement_items']}\n")
        f.write(f"Items with unanimous agreement: {disagreement_stats['unanimous_items']}\n")
        f.write(f"Percentage of items with disagreement: {disagreement_stats['disagreement_percentage']:.2f}%\n")
        f.write(f"Items used for agreement analysis: {disagreement_stats['disagreement_items']}\n")
        f.write("\n")
        
        # 0. Agreement Summary Statistics
        f.write("0. AGREEMENT SUMMARY STATISTICS\n")
        f.write("-" * 40 + "\n\n")
        
        f.write("Mean Agreement Values (computed only on disagreement items):\n")
        f.write(f"Intra-position agreement (diagonal):\n")
        f.write(f"  Raw Agreement: {agreement_summary['intra_raw_mean']:.4f} ± {agreement_summary['intra_raw_std']:.4f} (n={agreement_summary['intra_count']})\n")
        f.write(f"  Cohen's Kappa: {agreement_summary['intra_kappa_mean']:.4f} ± {agreement_summary['intra_kappa_std']:.4f} (n={agreement_summary['intra_count']})\n")
        f.write(f"  Gwet's AC1: {agreement_summary['intra_ac1_mean']:.4f} ± {agreement_summary['intra_ac1_std']:.4f} (n={agreement_summary['intra_count']})\n")
        f.write("\n")
        f.write(f"Inter-position agreement (off-diagonal):\n")
        f.write(f"  Raw Agreement: {agreement_summary['inter_raw_mean']:.4f} ± {agreement_summary['inter_raw_std']:.4f} (n={agreement_summary['inter_count']})\n")
        f.write(f"  Cohen's Kappa: {agreement_summary['inter_kappa_mean']:.4f} ± {agreement_summary['inter_kappa_std']:.4f} (n={agreement_summary['inter_count']})\n")
        f.write(f"  Gwet's AC1: {agreement_summary['inter_ac1_mean']:.4f} ± {agreement_summary['inter_ac1_std']:.4f} (n={agreement_summary['inter_count']})\n")
        f.write("\n")
        f.write(f"Overall agreement:\n")
        f.write(f"  Raw Agreement: {agreement_summary['overall_raw_mean']:.4f} ± {agreement_summary['overall_raw_std']:.4f} (n={agreement_summary['overall_count']})\n")
        f.write(f"  Cohen's Kappa: {agreement_summary['overall_kappa_mean']:.4f} ± {agreement_summary['overall_kappa_std']:.4f} (n={agreement_summary['overall_count']})\n")
        f.write(f"  Gwet's AC1: {agreement_summary['overall_ac1_mean']:.4f} ± {agreement_summary['overall_ac1_std']:.4f} (n={agreement_summary['overall_count']})\n")
        f.write("\n")

        # Comparison of Intra- vs. Inter-Position Agreement
        f.write("Comparison of Intra- vs. Inter-Position Agreement:\n")
        if agreement_summary.get('comparison_tests') and agreement_summary['comparison_tests']:
            for test in agreement_summary['comparison_tests']:
                f.write(f"  {test['label']}:\n")
                f.write(f"    Intra (μ={test['mean_1']:.4f}, n={test['n1']}) vs. Inter (μ={test['mean_2']:.4f}, n={test['n2']})\n")
                f.write(f"    Mann-Whitney U = {test['statistic']:.2f}, p = {test['p_value']:.4f}\n")
                f.write(f"    Cohen's d = {test['cohens_d']:.4f} (Effect: {test['effect_interpretation']})\n")
                f.write(f"    Significant (p < 0.05): {'Yes' if test['significant'] else 'No'}\n")
        else:
            f.write("  No comparison tests were performed between intra- and inter-position agreement.\n")
        f.write("\n\n")

        # 1. Agreement Matrices
        f.write("1. AGREEMENT MATRICES (Disagreement Items Only)\n")
        f.write("-" * 40 + "\n\n")
        
        f.write("Cohen's Kappa Agreement Matrix:\n")
        f.write(kappa_df.round(3).to_string())
        f.write("\n\n")
        
        f.write("Gwet's AC1 Agreement Matrix:\n")
        f.write(ac1_df.round(3).to_string())
        f.write("\n\n")
        
        f.write("Raw Agreement Matrix:\n")
        f.write(raw_agreement_df.round(3).to_string())
        f.write("\n\n")
        
        # 2. Descriptive Statistics
        f.write("2. DESCRIPTIVE STATISTICS BY POSITION (Disagreement Items Only)\n")
        f.write("-" * 40 + "\n\n")
        
        if descriptive_stats:
            # Group by position for better readability
            positions_in_data = sorted(list(set([stat['position'] for stat in descriptive_stats])))
            
            for pos in positions_in_data:
                pos_stats = [stat for stat in descriptive_stats if stat['position'] == pos]
                if pos_stats:
                    f.write(f"Position: {pos}\n")
                    f.write("-" * 20 + "\n")
                    
                    for stat in pos_stats:
                        f.write(f"  {stat['label']}:\n")
                        f.write(f"    n: {stat['n']}\n")
                        f.write(f"    Mean: {stat['mean']:.4f}\n")
                        f.write(f"    Std: {stat['std']:.4f}\n")
                        f.write(f"    Median: {stat['median']:.4f}\n")
                        f.write(f"    Min: {stat['min']:.4f}\n")
                        f.write(f"    Max: {stat['max']:.4f}\n")
                        f.write("\n")
                    f.write("\n")
        else:
            f.write("No descriptive statistics available.\n\n")
        
        # 3. Statistical Tests Summary
        f.write("3. STATISTICAL TESTS SUMMARY (Disagreement Items Only)\n")
        f.write("-" * 40 + "\n\n")
        
        if test_results:
            significant_tests = [t for t in test_results if t['significant']]
            f.write(f"Total comparisons: {len(test_results)}\n")
            f.write(f"Significant differences (p < 0.05): {len(significant_tests)}\n\n")
            
            if significant_tests:
                f.write("Significant differences found:\n")
                for test in significant_tests:
                    f.write(f"  {test['position_1']} vs {test['position_2']} ({test['label']}): "
                           f"p={test['p_value']:.4f}, d={test['cohens_d']:.3f} ({test['effect_interpretation']})\n")
                f.write("\n")
            else:
                f.write("No significant differences found.\n\n")
        else:
            f.write("No statistical tests performed.\n\n")
        
        # 4. Detailed Statistical Comparisons
        f.write("4. DETAILED STATISTICAL COMPARISONS (Disagreement Items Only)\n")
        f.write("-" * 40 + "\n\n")
        
        if test_results:
            # Group results by position comparison
            comparisons = {}
            for result in test_results:
                key = (result['position_1'], result['position_2'])
                if key not in comparisons:
                    comparisons[key] = []
                comparisons[key].append(result)
            
            # Write detailed comparisons
            for i, pos1 in enumerate(positions):
                for j, pos2 in enumerate(positions[i+1:], i+1):
                    key = (pos1, pos2)
                    if key in comparisons:
                        f.write(f"Comparing {pos1} vs {pos2}:\n")
                        f.write("-" * 30 + "\n")
                        
                        for result in comparisons[key]:
                            f.write(f"  {result['label']}:\n")
                            f.write(f"    {result['position_1']}: μ={result['mean_1']:.3f}, σ={result['std_1']:.3f}, n={result['n1']}\n")
                            f.write(f"    {result['position_2']}: μ={result['mean_2']:.3f}, σ={result['std_2']:.3f}, n={result['n2']}\n")
                            f.write(f"    Mann-Whitney U={result['statistic']:.2f}, p={result['p_value']:.4f}\n")
                            f.write(f"    Cohen's d={result['cohens_d']:.3f} ({result['effect_interpretation']})\n")
                            f.write(f"    Significant: {'Yes' if result['significant'] else 'No'}\n")
                            f.write("\n")
                        f.write("\n")
        else:
            f.write("No detailed comparisons available.\n\n")
        
        # 5. Statistical Test Results Table
        f.write("5. COMPLETE STATISTICAL TEST RESULTS TABLE (Disagreement Items Only)\n")
        f.write("-" * 40 + "\n\n")
        
        if test_results:
            # Create a formatted table
            df_tests = pd.DataFrame(test_results)
            df_tests = df_tests.round(4)
            
            # Format the table nicely
            f.write(df_tests.to_string(index=False))
            f.write("\n\n")
        else:
            f.write("No statistical test results available.\n\n")
        
        # 6. Effect Size Interpretation Guide
        f.write("6. EFFECT SIZE AND AGREEMENT INTERPRETATION GUIDE\n")
        f.write("-" * 40 + "\n\n")
        f.write("Cohen's d interpretation:\n")
        f.write("  < 0.2: negligible effect\n")
        f.write("  0.2-0.5: small effect\n")
        f.write("  0.5-0.8: medium effect\n")
        f.write("  > 0.8: large effect\n\n")
        
        f.write("Agreement coefficient interpretation:\n")
        f.write("  Cohen's κ: Chance-corrected agreement (affected by marginal distributions)\n")
        f.write("  Gwet's AC1: Chance-corrected agreement (less affected by marginal distributions)\n")
        f.write("  Raw Agreement: Simple proportion of exact matches\n\n")
        
        f.write("Statistical significance:\n")
        f.write("  p < 0.05: statistically significant\n")
        f.write("  p ≥ 0.05: not statistically significant\n\n")
        
        f.write("Important Note:\n")
        f.write("  All agreement calculations are performed only on items where personas\n")
        f.write("  disagree on the 'is_hate_speech' prediction. Items with unanimous\n")
        f.write("  agreement across all personas are excluded from the analysis.\n")
    
    print(f"Comprehensive statistical report saved to {output_path}")


def compute_agreement(labels, persona1_preds, persona2_preds, pos1=None, pos2=None):
    """Compute raw agreement, κ, and AC1 between two personas."""
    kappas = []
    ac1s = []
    raw_agreements = []
    perfect_agreement_count = 0
    
    for label in labels:
        y1 = persona1_preds[label].tolist()
        y2 = persona2_preds[label].tolist()
        
        kappa = safe_cohen_kappa(y1, y2, label)
        ac1 = safe_gwets_ac1(y1, y2, label)
        
        if kappa == 1.0:
            perfect_agreement_count += 1
            
        kappas.append(kappa)
        ac1s.append(ac1)

        # Raw agreement: proportion of exact matches
        matches = np.sum(np.array(y1) == np.array(y2))
        raw_agreement = matches / len(y1)
        raw_agreements.append(raw_agreement)

    return {
        "kappa": np.mean(kappas),
        "ac1": np.mean(ac1s),
        "raw_agreement": np.mean(raw_agreements),
        "perfect_agreements": perfect_agreement_count,
        "positions": (pos1, pos2)
    }

def compute_position_statistics(df, positions):

    # Get all columns after the first 4 metadata columns
    labels = list(df.columns[4:])  
    test_results = []
    descriptive_stats = []
    
    print("\n" + "="*50)
    print("STATISTICAL ANALYSIS (Disagreement Items Only)")
    print("="*50)
    
    # Identify only the numeric labels for statistical analysis
    numeric_labels = [label for label in labels if pd.api.types.is_numeric_dtype(df[label])]
    if not numeric_labels:
        print("Warning: No numeric columns found for statistical analysis.")
        return [], []

    print(f"Performing statistical analysis on numeric labels: {numeric_labels}")

    # Descriptive statistics for each position
    for pos in positions:
        pos_df = df[df["persona_pos"] == pos]
        
        for label in numeric_labels:  # Iterate only over numeric labels
            values = pos_df[label].values
            
            # Ensure there is data to process
            if len(values) == 0:
                continue

            desc_stat = {
                'position': pos,
                'label': label,
                'n': len(values),
                'mean': np.mean(values),
                'std': np.std(values, ddof=1),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            descriptive_stats.append(desc_stat)
    
    # Pairwise comparisons between positions
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions[i+1:], i+1):
            print(f"\nComparing {pos1} vs {pos2}:")
            
            pos1_df = df[df["persona_pos"] == pos1]
            pos2_df = df[df["persona_pos"] == pos2]
            
            for label in numeric_labels: # Iterate only over numeric labels
                group1 = pos1_df[label].values
                group2 = pos2_df[label].values
                
                if len(group1) == 0 or len(group2) == 0:
                    print(f"  {label}: No data for comparison")
                    continue
                
                test_result = perform_mann_whitney_test(group1, group2, label, pos1, pos2)
                if test_result:
                    test_results.append(test_result)
                    
                    # Print results
                    print(f"  {label}:")
                    print(f"    {pos1}: μ={test_result['mean_1']:.3f}, σ={test_result['std_1']:.3f}, n={test_result['n1']}")
                    print(f"    {pos2}: μ={test_result['mean_2']:.3f}, σ={test_result['std_2']:.3f}, n={test_result['n2']}")
                    print(f"    U={test_result['statistic']:.2f}, p={test_result['p_value']:.4f}")
                    print(f"    Cohen's d={test_result['cohens_d']:.3f} ({test_result['effect_interpretation']})")
                    print(f"    Significant: {'Yes' if test_result['significant'] else 'No'}")
    
    return descriptive_stats, test_results


def precompute_data_structures(df, positions, disagreement_items):
    """Pre-compute all necessary data structures for faster agreement computation, filtered by disagreement items"""
    labels = list(df.columns[4:])
    
    # Filter dataframe to only include disagreement items
    df_filtered = df[df['item_id'].isin(disagreement_items)]
    print(f"Filtered dataset size for agreement computation: {len(df_filtered)} rows (from {len(df)} total)")
    
    # Group data by position and persona for O(1) lookup
    position_data = {}
    persona_predictions = {}
    
    for pos in positions:
        pos_df = df_filtered[df_filtered["persona_pos"] == pos]
        position_data[pos] = {
            'df': pos_df,
            'personas': pos_df["persona_id"].unique() if not pos_df.empty else []
        }
        
        # Pre-compute predictions for each persona-label combination
        for persona in position_data[pos]['personas']:
            key = (pos, persona)
            persona_predictions[key] = {}
            persona_df = pos_df[pos_df["persona_id"] == persona]
            
            for label in labels:
                persona_predictions[key][label] = persona_df[label].values
    
    return labels, position_data, persona_predictions


def get_positions(task_config):
    # Determine position types based on classification path
    is_corners = "corners" in task_config["extreme_pos_path"]

    if is_corners:
        positions = ["top_left", "top_right", "bottom_left", "bottom_right"]
    else:
        positions = ["leftmost", "rightmost"]

    return positions


def safe_cohen_kappa(y1, y2, label_type):
    # If both arrays contain only one unique value and they're equal
    if len(set(y1)) == 1 and len(set(y2)) == 1 and set(y1) == set(y2):
        return 1.0

    try:
        return cohen_kappa_score(
            y1, y2
        )
    except Exception as e:
        print(f"Error computing kappa for {label_type}: {str(e)}")
        return np.nan


def safe_gwets_ac1(y1, y2, label_type):
    """Safely compute Gwet's AC1 with error handling"""
    # If both arrays contain only one unique value and they're equal
    if len(set(y1)) == 1 and len(set(y2)) == 1 and set(y1) == set(y2):
        return 1.0

    try:
        return gwets_ac1(y1, y2)
    except Exception as e:
        print(f"Error computing AC1 for {label_type}: {str(e)}")
        return np.nan

def compute_inter_agreement(pos1, pos2, labels, position_data, persona_predictions, pairwise_results):
    """Optimized inter-agreement computation using pre-computed data and compute_agreement function"""
    print(f"\nComputing inter-agreement between {pos1} and {pos2}")
    
    # Quick check for empty data
    if not position_data[pos1]['personas'].size or not position_data[pos2]['personas'].size:
        print(f"Warning: No data for position {pos1} or {pos2}")
        return np.nan, np.nan, np.nan, []
    
    personas1 = position_data[pos1]['personas']
    personas2 = position_data[pos2]['personas']
    
    all_kappas = []
    all_ac1s = []
    all_raw_agreements = []
    perfect_agreement_details = []
    
    # Use pre-computed predictions - no DataFrame filtering in inner loop
    for p1 in personas1:
        p1_preds = {label: persona_predictions[(pos1, p1)][label] for label in labels}
        
        for p2 in personas2:
            p2_preds = {label: persona_predictions[(pos2, p2)][label] for label in labels}
            
            # Use the compute_agreement function
            agreement_result = compute_agreement(labels, p1_preds, p2_preds, pos1, pos2)
            
            kappa = agreement_result["kappa"]
            ac1 = agreement_result["ac1"]
            raw_agreement = agreement_result["raw_agreement"]
            perfect_count = agreement_result["perfect_agreements"]
            
            if perfect_count > 0:
                perfect_agreement_details.append({
                    'pos1': pos1, 'pos2': pos2, 'count': perfect_count,
                    'personas': (p1, p2)
                })
            
            if not np.isnan(kappa):
                all_kappas.append(kappa)
                all_ac1s.append(ac1)
                all_raw_agreements.append(raw_agreement)
                
            pairwise_results.append((p1, p2, kappa, ac1, raw_agreement))
    
    print(f"Done processing inter-agreement between positions {pos1} and {pos2}")
    return (np.mean(all_kappas) if all_kappas else np.nan, 
            np.mean(all_ac1s) if all_ac1s else np.nan,
            np.mean(all_raw_agreements) if all_raw_agreements else np.nan,
            perfect_agreement_details)


def compute_intra_agreement(pos, labels, position_data, persona_predictions, pairwise_results):
    """Optimized intra-agreement computation using pre-computed data and compute_agreement function"""
    print(f"\nComputing intra-agreement for {pos}")
    
    # Quick check for empty data
    if not position_data[pos]['personas'].size:
        print(f"Warning: No data for position {pos}")
        return np.nan, np.nan, np.nan, []
    
    personas = position_data[pos]['personas']
    all_kappas = []
    all_ac1s = []
    all_raw_agreements = []
    perfect_agreement_details = []
    
    # Use pre-computed predictions - no DataFrame filtering in inner loop
    for i in range(len(personas)):
        p1 = personas[i]
        p1_preds = {label: persona_predictions[(pos, p1)][label] for label in labels}
        
        for j in range(i + 1, len(personas)):
            p2 = personas[j]
            p2_preds = {label: persona_predictions[(pos, p2)][label] for label in labels}
            
            # Use the compute_agreement function
            agreement_result = compute_agreement(labels, p1_preds, p2_preds, pos, pos)
            
            kappa = agreement_result["kappa"]
            ac1 = agreement_result["ac1"]
            raw_agreement = agreement_result["raw_agreement"]
            perfect_count = agreement_result["perfect_agreements"]
            
            if perfect_count > 0:
                perfect_agreement_details.append({
                    'pos1': pos, 'pos2': pos, 'count': perfect_count,
                    'personas': (p1, p2)
                })
            
            if not np.isnan(kappa):
                all_kappas.append(kappa)
                all_ac1s.append(ac1)
                all_raw_agreements.append(raw_agreement)
            else:
                print("ERROR!!!!")
                
            pairwise_results.append((p1, p2, kappa, ac1, raw_agreement))
    
    print(f"Done processing intra-agreement for position {pos}")
    return (np.mean(all_kappas) if all_kappas else np.nan,
            np.mean(all_ac1s) if all_ac1s else np.nan,
            np.mean(all_raw_agreements) if all_raw_agreements else np.nan,
            perfect_agreement_details)


# Wrapper functions for multiprocessing (must be picklable)
def compute_agreement_task(args):
    """Worker function that computes agreement for a single position pair"""
    i, j, pos1, pos2, labels, position_data, persona_predictions = args
    
    print(f"\nProcessing position pair ({pos1}, {pos2})")
    
    # Store individual agreement values for proper mean calculation
    individual_agreements = []
    
    if i == j:
        kappa_result, ac1_result, raw_agreement_result, perfect_agreements = compute_intra_agreement(
            pos1, labels, position_data, persona_predictions, individual_agreements
        )
    else:
        kappa_result, ac1_result, raw_agreement_result, perfect_agreements = compute_inter_agreement(
            pos1, pos2, labels, position_data, persona_predictions, individual_agreements
        )
    
    # Extract individual agreement values from the stored results
    individual_kappas = [agreement[2] for agreement in individual_agreements if not np.isnan(agreement[2])]
    individual_ac1s = [agreement[3] for agreement in individual_agreements if not np.isnan(agreement[3])]
    individual_raws = [agreement[4] for agreement in individual_agreements if not np.isnan(agreement[4])]
    
    return (i, j, kappa_result, ac1_result, raw_agreement_result, perfect_agreements, 
            individual_kappas, individual_ac1s, individual_raws)


def compute_agreement_summary(results, positions):
    """Compute proper mean agreement values for intra vs inter comparisons"""
    
    # Collect all individual agreement values
    all_intra_kappas = []
    all_intra_ac1s = []
    all_intra_raws = []
    
    all_inter_kappas = []
    all_inter_ac1s = []
    all_inter_raws = []
    
    for result in results:
        i, j, _, _, _, _, individual_kappas, individual_ac1s, individual_raws = result
        
        if i == j:  # Diagonal (intra-position)
            all_intra_kappas.extend(individual_kappas)
            all_intra_ac1s.extend(individual_ac1s)
            all_intra_raws.extend(individual_raws)
        else:  # Off-diagonal (inter-position)
            all_inter_kappas.extend(individual_kappas)
            all_inter_ac1s.extend(individual_ac1s)
            all_inter_raws.extend(individual_raws)
    
    # Combine all for overall statistics
    all_kappas = all_intra_kappas + all_inter_kappas
    all_ac1s = all_intra_ac1s + all_inter_ac1s
    all_raws = all_intra_raws + all_inter_raws
    
    # Compute summary statistics
    summary = {
        'intra_kappa_mean': np.mean(all_intra_kappas) if all_intra_kappas else np.nan,
        'intra_kappa_std': np.std(all_intra_kappas) if all_intra_kappas else np.nan,
        'intra_ac1_mean': np.mean(all_intra_ac1s) if all_intra_ac1s else np.nan,
        'intra_ac1_std': np.std(all_intra_ac1s) if all_intra_ac1s else np.nan,
        'intra_raw_mean': np.mean(all_intra_raws) if all_intra_raws else np.nan,
        'intra_raw_std': np.std(all_intra_raws) if all_intra_raws else np.nan,
        'intra_count': len(all_intra_kappas),
        
        'inter_kappa_mean': np.mean(all_inter_kappas) if all_inter_kappas else np.nan,
        'inter_kappa_std': np.std(all_inter_kappas) if all_inter_kappas else np.nan,
        'inter_ac1_mean': np.mean(all_inter_ac1s) if all_inter_ac1s else np.nan,
        'inter_ac1_std': np.std(all_inter_ac1s) if all_inter_ac1s else np.nan,
        'inter_raw_mean': np.mean(all_inter_raws) if all_inter_raws else np.nan,
        'inter_raw_std': np.std(all_inter_raws) if all_inter_raws else np.nan,
        'inter_count': len(all_inter_kappas),
        
        'overall_kappa_mean': np.mean(all_kappas) if all_kappas else np.nan,
        'overall_kappa_std': np.std(all_kappas) if all_kappas else np.nan,
        'overall_ac1_mean': np.mean(all_ac1s) if all_ac1s else np.nan,
        'overall_ac1_std': np.std(all_ac1s) if all_ac1s else np.nan,
        'overall_raw_mean': np.mean(all_raws) if all_raws else np.nan,
        'overall_raw_std': np.std(all_raws) if all_raws else np.nan,
        'overall_count': len(all_kappas)
    }

    # Perform statistical comparison between intra- and inter-position agreements
    comparison_tests = []
    
    # Raw Agreement
    if all_intra_raws and all_inter_raws:
        test_raw = perform_mann_whitney_test(
            all_intra_raws, all_inter_raws, 
            'Raw Agreement', 'Intra-Position', 'Inter-Position'
        )
        if test_raw:
            comparison_tests.append(test_raw)
            
    # Cohen's Kappa
    if all_intra_kappas and all_inter_kappas:
        test_kappa = perform_mann_whitney_test(
            all_intra_kappas, all_inter_kappas, 
            "Cohen's Kappa", 'Intra-Position', 'Inter-Position'
        )
        if test_kappa:
            comparison_tests.append(test_kappa)
            
    # Gwet's AC1
    if all_intra_ac1s and all_inter_ac1s:
        test_ac1 = perform_mann_whitney_test(
            all_intra_ac1s, all_inter_ac1s, 
            "Gwet's AC1", 'Intra-Position', 'Inter-Position'
        )
        if test_ac1:
            comparison_tests.append(test_ac1)
            
    summary['comparison_tests'] = comparison_tests
    
    return summary


def main():
    global DATASET_NAME

    parser = argparse.ArgumentParser(description="Run content classification and statistical analysis pipeline for multiple result files.")
    parser.add_argument(
        "--input_paths",
        type=str,
        nargs='+',  # This allows for one or more arguments
        default=[
            "data/results/img_classification/Idefics3-8B-Llama3/20250813_162245.json/final_results.json", # Facebook
            "data/results/text_classification/Llama-3.1-8B-Instruct/20250711_112214/final_results.json", # yoder 4-corner
            "data/results/text_classification/Llama-3.1-8B-Instruct/20250713_013923/final_results.json", # subdata l-r
            "data/results/text_classification/Llama-3.1-70B-Instruct/20250724_140425/final_results.json", # yoder 4-corner
            "data/results/text_classification/Llama-3.1-70B-Instruct/20250717_084101/final_results.json", # subdata l-r
            "data/results/text_classification/Qwen2.5-32B-Instruct/20250717_215350/final_results.json", # yoder 4-corner
            "data/results/text_classification/Qwen2.5-32B-Instruct/20250713_103728/final_results.json", # subdata l-r
            "data/results/img_classification/Qwen2.5-VL-7B-Instruct/20250712_012750.json/final_results.json", # Facebook
            "data/results/img_classification/Qwen2.5-VL-32B-Instruct/20250712_030201/final_results.json", # Facebook
        ],
        help="One or more paths to the final_results.json files to process."
    )
    args = parser.parse_args()

    for input_path in args.input_paths:
        print(f"\n{'='*30} PROCESSING FILE: {input_path} {'='*30}\n")
        try:
            with open(input_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading file {input_path}: {str(e)}. Skipping.")
            continue
        
        task_config = data["metadata"]["task_config"]
        model_config = data["metadata"]["model_config"]

        MODEL_NAME = model_config["name"].split("/")[-1]
        TIMESTAMP = input_path.split("/")[-2]
        DATASET_NAME = data["metadata"]["dataset_name"]

        first_result_labels = data["results"][0]["true_labels"]
        first_result_labels = list(first_result_labels.keys())
        is_harmful_label = first_result_labels[0]

        # Define output path for comprehensive report, unique for each input file
        comprehensive_report_path = f"images/agreement/{MODEL_NAME}/comprehensive_report_disagreement_only_no_atk_mtd_{TIMESTAMP}.txt"

        predictions = data["results"]
        df = pd.DataFrame(predictions)

        print(df.columns)
        print()
        print("=" * 70)
        print(f"\nDataset size: {len(df)} rows")
        print(f"Unique items: {df['item_id'].nunique()}")
        print(f"Positions present: {df['persona_pos'].unique()}")
        print(f"Personas present: {df['persona_id'].nunique()}")
        print("=" * 70)
        print()

        df["is_hate_speech"] = df["predicted_labels"].apply(lambda x: int(x[is_harmful_label]))

        if len(df["predicted_labels"].iloc[0]) > 1:
            sample_labels = df["predicted_labels"].iloc[0]
            other_columns = [col for col in sample_labels.keys() if col != is_harmful_label]
            print(f"Other columns found: {other_columns}")
            
            if other_columns:
                first_col = other_columns[0]
                df[first_col] = df["predicted_labels"].apply(lambda x: x[first_col])
                print(f"Added column: {first_col}")

        df = df.drop(["predicted_labels", "true_labels"], axis=1)

        positions = get_positions(task_config)
        
        # NEW: Identify disagreement items
        disagreement_items = identify_disagreement_items(df, positions)
        
        # Store disagreement statistics for the report
        disagreement_stats = {
            'total_items': df['item_id'].nunique(),
            'disagreement_items': len(disagreement_items),
            'unanimous_items': df['item_id'].nunique() - len(disagreement_items),
            'disagreement_percentage': len(disagreement_items) / df['item_id'].nunique() * 100
        }
        
        # If no disagreement items, skip this file
        if len(disagreement_items) == 0:
            print(f"Warning: No disagreement items found for {input_path}. Skipping agreement analysis.")
            continue
        
        # Filter dataframe for statistical analysis (only disagreement items)
        df_disagreement = df[df['item_id'].isin(disagreement_items)]
        
        matrix_size = len(positions)
        
        # Compute position statistics on disagreement items only
        descriptive_stats, test_results = compute_position_statistics(df_disagreement, positions)
        
        kappa_matrix = np.zeros((matrix_size, matrix_size))
        ac1_matrix = np.zeros((matrix_size, matrix_size))
        raw_agreement_matrix = np.zeros((matrix_size, matrix_size))
        
        # Pre-compute data structures with disagreement items only
        labels, position_data, persona_predictions = precompute_data_structures(df, positions, disagreement_items)

        tasks = []
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions[i:], i):
                tasks.append((i, j, pos1, pos2, labels, position_data, persona_predictions))

        num_processes = min(len(tasks), mp.cpu_count())
        print(f"Running {len(tasks)} tasks on {num_processes} processes for {input_path}")
        
        with mp.Pool(num_processes) as pool:
            results = pool.map(compute_agreement_task, tasks)
        
        all_perfect_agreements = []
        for result in results:
            i, j, kappa_result, ac1_result, raw_agreement_result, perfect_agreements = result[:6]
            kappa_matrix[i, j] = kappa_result
            ac1_matrix[i, j] = ac1_result
            raw_agreement_matrix[i, j] = raw_agreement_result
            all_perfect_agreements.extend(perfect_agreements)
            if i != j:
                kappa_matrix[j, i] = kappa_result
                ac1_matrix[j, i] = ac1_result
                raw_agreement_matrix[j, i] = raw_agreement_result

        agreement_summary = compute_agreement_summary(results, positions)
        
        print(f"\n" + "="*65)
        print(f"RESULTS FOR: {input_path}")
        print("DISAGREEMENT FILTERING RESULTS")
        print("="*65)
        print(f"Total items: {disagreement_stats['total_items']}")
        print(f"Items with disagreement: {disagreement_stats['disagreement_items']}")
        print(f"Items with unanimous agreement: {disagreement_stats['unanimous_items']}")
        print(f"Percentage with disagreement: {disagreement_stats['disagreement_percentage']:.2f}%")
        
        print(f"\n" + "="*65)
        print("AGREEMENT SUMMARY STATISTICS (Disagreement Items Only)")
        print("="*65)
        print(f"Intra-position agreement (n={agreement_summary['intra_count']}):")
        print(f"  Raw Agreement: {agreement_summary['intra_raw_mean']:.4f} ± {agreement_summary['intra_raw_std']:.4f}")
        print(f"  Cohen's Kappa: {agreement_summary['intra_kappa_mean']:.4f} ± {agreement_summary['intra_kappa_std']:.4f}")
        print(f"  Gwet's AC1: {agreement_summary['intra_ac1_mean']:.4f} ± {agreement_summary['intra_ac1_std']:.4f}")
        print(f"Inter-position agreement (n={agreement_summary['inter_count']}):")
        print(f"  Raw Agreement: {agreement_summary['inter_raw_mean']:.4f} ± {agreement_summary['inter_raw_std']:.4f}")
        print(f"  Cohen's Kappa: {agreement_summary['inter_kappa_mean']:.4f} ± {agreement_summary['inter_kappa_std']:.4f}")
        print(f"  Gwet's AC1: {agreement_summary['inter_ac1_mean']:.4f} ± {agreement_summary['inter_ac1_std']:.4f}")
        print(f"Overall agreement (n={agreement_summary['overall_count']}):")
        print(f"  Raw Agreement: {agreement_summary['overall_raw_mean']:.4f} ± {agreement_summary['overall_raw_std']:.4f}")
        print(f"  Cohen's Kappa: {agreement_summary['overall_kappa_mean']:.4f} ± {agreement_summary['overall_kappa_std']:.4f}")
        print(f"  Gwet's AC1: {agreement_summary['overall_ac1_mean']:.4f} ± {agreement_summary['overall_ac1_std']:.4f}")

        if agreement_summary.get('comparison_tests'):
            print("\n" + "="*65)
            print("INTRA- VS. INTER-POSITION AGREEMENT COMPARISON (Disagreement Items Only)")
            print("="*65)
            for test in agreement_summary['comparison_tests']:
                print(f"{test['label']}:")
                print(f"  Intra (n={test['n1']}, μ={test['mean_1']:.3f}) vs. Inter (n={test['n2']}, μ={test['mean_2']:.3f})")
                print(f"  Mann-Whitney U={test['statistic']:.2f}, p={test['p_value']:.4f}")
                print(f"  Cohen's d={test['cohens_d']:.3f} ({test['effect_interpretation']}) - {'Significant' if test['significant'] else 'Not Significant'}\n")

        perfect_agreement_counts = {pos: 0 for pos in positions}
        for agreement in all_perfect_agreements:
            pos1, pos2, count = agreement['pos1'], agreement['pos2'], agreement['count']
            if pos1 == pos2:
                perfect_agreement_counts[pos1] += count
            else:
                perfect_agreement_counts[pos1] += count * 0.5
                perfect_agreement_counts[pos2] += count * 0.5

        print(f"\nPerfect Agreement Counts by Position (Disagreement Items Only):")
        print("-" * 65)
        for pos in positions:
            print(f"{pos}: {perfect_agreement_counts[pos]} exact matches")
        print(f"Total exact matches: {sum(perfect_agreement_counts.values())}")

        kappa_df = pd.DataFrame(kappa_matrix, index=positions, columns=positions)
        ac1_df = pd.DataFrame(ac1_matrix, index=positions, columns=positions)
        raw_agreement_df = pd.DataFrame(raw_agreement_matrix, index=positions, columns=positions)

        output_dir = os.path.dirname(comprehensive_report_path)
        os.makedirs(output_dir, exist_ok=True)

        save_comprehensive_report(kappa_df, ac1_df, raw_agreement_df, descriptive_stats, test_results, positions, comprehensive_report_path, agreement_summary, disagreement_stats)
    

if __name__ == "__main__":
    main()