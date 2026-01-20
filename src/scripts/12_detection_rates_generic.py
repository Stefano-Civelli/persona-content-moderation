import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact
from itertools import combinations
import os
from collections import defaultdict
import warnings
from contextlib import redirect_stdout
import sys

warnings.filterwarnings('ignore')

def load_predictions(file_path):
    """Load predictions from JSON file"""
    print(f"Loading predictions from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract results
    results = data['results']
    df = pd.DataFrame(results)
    
    print(f"Loaded {len(df)} predictions")
    return df

def analyze_persona_bias(df, output_dir="images/subdata"):
    """Analyze if personas from different positions show bias in detecting harmful content"""
    
    # Basic statistics
    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    
    print(f"Total samples: {len(df)}")
    print(f"Unique personas: {df['persona_id'].nunique()}")
    
    # Check available positions
    available_positions = sorted(df['persona_pos'].unique())
    print(f"Available persona positions: {list(available_positions)}")
    
    # Extract relevant columns
    df_clean = df.copy()
    df_clean['true_is_hate'] = df_clean['true_labels'].apply(lambda x: x.get('is_hate_speech') if isinstance(x, dict) else None)
    df_clean['pred_is_hate'] = df_clean['predicted_labels'].apply(lambda x: x.get('is_hate_speech') if isinstance(x, dict) else None)
    df_clean['target_category'] = df_clean['true_labels'].apply(lambda x: x.get('target_category') if isinstance(x, dict) else None)
    
    # Automatically detect available target categories (excluding None values)
    available_targets = [t for t in df_clean['target_category'].unique() if t is not None]
    available_targets = sorted(available_targets)  # Sort for consistent ordering
    print(f"Available target categories: {available_targets}")
    
    if not available_targets:
        print("WARNING: No target categories found in the data!")
        return None
    
    # Count samples by true label status
    true_hate_count = df_clean['true_is_hate'].sum() if df_clean['true_is_hate'].notna().any() else 0
    true_non_hate_count = (df_clean['true_is_hate'] == False).sum() if df_clean['true_is_hate'].notna().any() else 0
    missing_true_labels = df_clean['true_is_hate'].isna().sum()
    
    print(f"\nTrue hate speech samples: {true_hate_count}")
    print(f"True non-hate speech samples: {true_non_hate_count}")
    if missing_true_labels > 0:
        print(f"Samples with missing true labels: {missing_true_labels}")
    
    # Count predicted labels
    pred_hate_count = df_clean['pred_is_hate'].sum() if df_clean['pred_is_hate'].notna().any() else 0
    pred_non_hate_count = (df_clean['pred_is_hate'] == False).sum() if df_clean['pred_is_hate'].notna().any() else 0
    missing_pred_labels = df_clean['pred_is_hate'].isna().sum()
    
    print(f"Predicted as hate speech: {pred_hate_count}")
    print(f"Predicted as non-hate speech: {pred_non_hate_count}")
    if missing_pred_labels > 0:
        print(f"Samples with missing predictions: {missing_pred_labels}")
    
    # Filter out samples with missing predictions
    #df_analysis = df_clean[df_clean['pred_is_hate'].notna()].copy()

    # Filter out samples with missing predictions AND only keep true hate speech samples
    df_analysis = df_clean[(df_clean['pred_is_hate'].notna()) & (df_clean['true_is_hate'] == True)].copy()
    print(f"\nSamples available for analysis: {len(df_analysis)}")
    
    if len(df_analysis) == 0:
        print("ERROR: No samples with valid predictions found!")
        return None
    
    # Analysis 1: Overall detection rates by position
    print("\n" + "="*60)
    print("OVERALL DETECTION RATES BY POSITION")
    print("="*60)
    print("(Detection rate = proportion of samples flagged as harmful)")
    
    detection_rates = {}
    for pos in available_positions:
        pos_data = df_analysis[df_analysis['persona_pos'] == pos]
        if len(pos_data) > 0:
            detection_rate = pos_data['pred_is_hate'].mean()
            detection_rates[pos] = detection_rate
            print(f"{pos}: {detection_rate:.3f} ({pos_data['pred_is_hate'].sum()}/{len(pos_data)})")
    
    # Visualize overall detection rates
    if detection_rates:
        plt.figure(figsize=(10, 6))
        positions = list(detection_rates.keys())
        rates = list(detection_rates.values())
        
        # Use a color palette that works for any number of positions
        colors = plt.cm.Set3(np.linspace(0, 1, len(positions)))
        
        bars = plt.bar(positions, rates, color=colors)
        plt.title('Harmful Content Detection Rates by Persona Position', fontsize=14, fontweight='bold')
        plt.xlabel('Persona Position', fontsize=12)
        plt.ylabel('Detection Rate (Proportion Flagged as Harmful)', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overall_detection_rates.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Analysis 2: Detection rates by target category and position
    print("\n" + "="*60)
    print("DETECTION RATES BY TARGET CATEGORY AND POSITION")
    print("="*60)
    
    category_position_stats = defaultdict(lambda: defaultdict(dict))
    
    for target in available_targets:
        print(f"\n--- Target Category: {target} ---")
        target_data = df_analysis[df_analysis['target_category'] == target]
        
        if len(target_data) == 0:
            print(f"No data for target category: {target}")
            continue
            
        category_stats = {}
        for pos in available_positions:
            pos_target_data = target_data[target_data['persona_pos'] == pos]
            if len(pos_target_data) > 0:
                detection_rate = pos_target_data['pred_is_hate'].mean()
                total_samples = len(pos_target_data)
                detected_samples = pos_target_data['pred_is_hate'].sum()
                
                category_stats[pos] = {
                    'rate': detection_rate,
                    'detected': detected_samples,
                    'total': total_samples
                }
                
                print(f"  {pos}: {detection_rate:.3f} ({detected_samples}/{total_samples})")
        
        category_position_stats[target] = category_stats
    
    # Create heatmap of detection rates
    if available_targets and available_positions:
        heatmap_data = []
        for target in available_targets:
            row = []
            for pos in available_positions:
                if pos in category_position_stats[target]:
                    row.append(category_position_stats[target][pos]['rate'])
                else:
                    row.append(np.nan)
            heatmap_data.append(row)
        
        plt.figure(figsize=(max(12, len(available_positions) * 2), max(8, len(available_targets) * 0.8)))
        heatmap_df = pd.DataFrame(heatmap_data, index=available_targets, columns=available_positions)
        sns.heatmap(heatmap_df, annot=True, cmap='YlOrRd', center=0.5, 
                    fmt='.3f', cbar_kws={'label': 'Detection Rate'})
        plt.title('Harmful Content Detection Rates by Target Category and Persona Position', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Persona Position', fontsize=12)
        plt.ylabel('Target Category', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/detection_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Analysis 3: Statistical significance tests
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*60)
    
    significant_results = []
    
    for target in available_targets:
        print(f"\n--- Statistical Tests for Target: {target} ---")
        target_data = df_analysis[df_analysis['target_category'] == target]
        
        if len(target_data) < 10:
            print(f"Insufficient data for statistical testing (n={len(target_data)})")
            continue
        
        # Create contingency table for chi-square test
        contingency_data = []
        position_labels = []
        
        for pos in available_positions:
            pos_data = target_data[target_data['persona_pos'] == pos]
            if len(pos_data) >= 5:  # Minimum sample size for chi-square
                detected = pos_data['pred_is_hate'].sum()
                not_detected = len(pos_data) - detected
                contingency_data.append([detected, not_detected])
                position_labels.append(pos)
        
        if len(contingency_data) >= 2:
            contingency_table = np.array(contingency_data)
            
            # Chi-square test
            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                print(f"  Overall Chi-square test: χ² = {chi2:.4f}, p = {p_value:.4f}")
                
                if p_value < 0.05:
                    print(f"  *** SIGNIFICANT OVERALL DIFFERENCE FOUND (p < 0.05) ***")
                    significant_results.append({
                        'target': target,
                        'test': 'chi-square',
                        'statistic': chi2,
                        'p_value': p_value,
                        'positions': position_labels
                    })
                else:
                    print(f"  No significant difference found")
            except ValueError as e:
                print(f"  Unable to perform Chi-square test: {e}")
                continue
            
            # Pairwise comparisons with Bonferroni correction
            print(f"\n  Pairwise comparisons (Fisher's exact test):")
            
            pos_to_idx = {pos: i for i, pos in enumerate(position_labels)}
            pairwise_comparisons = list(combinations(position_labels, 2))
            num_comparisons = len(pairwise_comparisons)
            
            if num_comparisons > 0:
                bonferroni_alpha = 0.05 / num_comparisons
                print(f"    Applying Bonferroni correction for {num_comparisons} tests. New alpha = {bonferroni_alpha:.4f}")

                for pos1, pos2 in pairwise_comparisons:
                    table_2x2 = np.array([
                        contingency_table[pos_to_idx[pos1]],
                        contingency_table[pos_to_idx[pos2]]
                    ])
                    
                    try:
                        odds_ratio, p_fisher = fisher_exact(table_2x2)
                        
                        if p_fisher < bonferroni_alpha:
                            significance_marker = f"*** SIGNIFICANT (p < {bonferroni_alpha:.4f}) ***"
                            significant_results.append({
                                'target': target,
                                'test': 'fisher_exact',
                                'comparison': f"{pos1} vs {pos2}",
                                'odds_ratio': odds_ratio, 
                                'p_value': p_fisher,
                                'bonferroni_alpha': bonferroni_alpha
                            })
                        else:
                            significance_marker = ""
                            
                        print(f"      {pos1} vs {pos2}: OR = {odds_ratio:.3f}, p = {p_fisher:.4f} {significance_marker}")

                    except ValueError as e:
                        print(f"      {pos1} vs {pos2}: Unable to compute Fisher's exact test: {e}")
    
    # Create visualization of significant results
    if significant_results:
        print("\n" + "="*60)
        print("SUMMARY OF SIGNIFICANT RESULTS")
        print("="*60)
        
        target_sig_results = defaultdict(list)
        for result in significant_results:
            target_sig_results[result['target']].append(result)
        
        # Create bar plot of p-values for significant results
        fig, axes = plt.subplots(len(target_sig_results), 1, 
                                figsize=(12, 4 * len(target_sig_results)))
        
        if len(target_sig_results) == 1:
            axes = [axes]
        
        for idx, (target, results) in enumerate(target_sig_results.items()):
            ax = axes[idx] if len(target_sig_results) > 1 else axes[0]
            
            labels = []
            p_values = []
            colors = []
            
            for result in results:
                if result['test'] == 'chi-square':
                    labels.append(f"Overall\n(χ²)")
                    colors.append('#FF6B6B')
                else:
                    labels.append(f"{result['comparison']}\n(Fisher)")
                    colors.append('#4ECDC4')
                p_values.append(result['p_value'])
            
            bars = ax.bar(range(len(labels)), p_values, color=colors)
            ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Standard α = 0.05')
            
            # Add lines for Bonferroni thresholds if they exist
            bonf_alphas = {r['bonferroni_alpha'] for r in results if 'bonferroni_alpha' in r}
            for i, alpha in enumerate(bonf_alphas):
                ax.axhline(y=alpha, color='purple', linestyle=':', alpha=0.8, 
                          label=f'Bonferroni α ≈ {alpha:.3f}' if i==0 else "")

            ax.set_title(f'Significant P-Values for Target: {target}', fontweight='bold')
            ax.set_xlabel('Comparison')
            ax.set_ylabel('p-value')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.legend()
            
            # Add value labels on bars
            for bar, p_val in zip(bars, p_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                       f'{p_val:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/significant_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary
        for target, results in target_sig_results.items():
            print(f"\n{target}:")
            for result in results:
                if result['test'] == 'chi-square':
                    print(f"  - Overall difference: p = {result['p_value']:.4f}")
                else:
                    print(f"  - {result['comparison']}: OR = {result['odds_ratio']:.3f}, p = {result['p_value']:.4f} (Bonferroni α = {result['bonferroni_alpha']:.4f})")
    
    else:
        print("\nNo statistically significant differences found after corrections.")
    
    # Analysis 4: Effect sizes and confidence intervals
    print("\n" + "="*60)
    print("EFFECT SIZES AND PRACTICAL SIGNIFICANCE")
    print("="*60)
    
    effect_sizes = []
    for target in available_targets:
        target_data = df_analysis[df_analysis['target_category'] == target]
        
        if len(target_data) < 10:
            continue
            
        print(f"\n--- Effect Sizes for Target: {target} ---")
        
        # Calculate effect sizes between positions
        for pos1, pos2 in combinations(available_positions, 2):
            pos1_data = target_data[target_data['persona_pos'] == pos1]
            pos2_data = target_data[target_data['persona_pos'] == pos2]
            
            if len(pos1_data) >= 5 and len(pos2_data) >= 5:
                rate1 = pos1_data['pred_is_hate'].mean()
                rate2 = pos2_data['pred_is_hate'].mean()
                
                # Cohen's h for proportions
                h = 2 * (np.arcsin(np.sqrt(rate1)) - np.arcsin(np.sqrt(rate2)))
                
                print(f"  {pos1} vs {pos2}:")
                print(f"    Detection rates: {rate1:.3f} vs {rate2:.3f} (diff: {rate1-rate2:.3f})")
                print(f"    Cohen's h: {h:.3f} ", end="")
                
                if abs(h) < 0.2:
                    print("(small effect)")
                elif abs(h) < 0.5:
                    print("(medium effect)")
                else:
                    print("(large effect)")
                
                effect_sizes.append({
                    'target': target,
                    'comparison': f"{pos1} vs {pos2}",
                    'rate1': rate1,
                    'rate2': rate2,
                    'difference': rate1 - rate2,
                    'cohens_h': h
                })
    
    # Create effect size visualization
    if effect_sizes:
        df_effects = pd.DataFrame(effect_sizes)
        targets_with_effects = df_effects['target'].unique()
        
        fig, axes = plt.subplots(len(targets_with_effects), 1, 
                                figsize=(12, 4 * len(targets_with_effects)))
        
        if len(targets_with_effects) == 1:
            axes = [axes]
        
        for idx, target in enumerate(targets_with_effects):
            ax = axes[idx] if len(targets_with_effects) > 1 else axes[0]
            
            target_effects = df_effects[df_effects['target'] == target]
            
            # Color code by effect size magnitude
            colors = ['#FF6B6B' if abs(h) >= 0.5 else '#FFD93D' if abs(h) >= 0.2 else '#4ECDC4' 
                     for h in target_effects['cohens_h']]
            
            bars = ax.bar(range(len(target_effects)), target_effects['cohens_h'], color=colors)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Small effect (h=0.2)')
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Medium effect (h=0.5)')
            ax.axhline(y=-0.2, color='orange', linestyle='--', alpha=0.5)
            ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5)
            
            ax.set_title(f"Effect Sizes (Cohen's h) for Target: {target}", fontweight='bold')
            ax.set_xlabel('Comparison')
            ax.set_ylabel("Cohen's h")
            ax.set_xticks(range(len(target_effects)))
            ax.set_xticklabels(target_effects['comparison'], rotation=45, ha='right')
            ax.legend()
            
            # Add value labels
            for bar, h in zip(bars, target_effects['cohens_h']):
                ax.text(bar.get_x() + bar.get_width()/2, 
                       bar.get_height() + (0.02 if bar.get_height() >= 0 else -0.05), 
                       f'{h:.3f}', ha='center', va='bottom' if bar.get_height() >= 0 else 'top', 
                       fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/effect_sizes.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}/")
    print("Generated plots:")
    print("- overall_detection_rates.png: Detection rates by persona position")
    print("- detection_heatmap.png: Detection rates by target category and position")
    if significant_results:
        print("- significant_results.png: Statistical significance results (with Bonferroni correction)")
    if effect_sizes:
        print("- effect_sizes.png: Effect sizes between positions")
    
    return {
        'detection_rates': detection_rates,
        'category_position_stats': dict(category_position_stats),
        'significant_results': significant_results,
        'effect_sizes': effect_sizes
    }

if __name__ == "__main__":
    file_path = "paper_results/Qwen2.5-32B-Instruct/20250722_125943/final_results.json"
    file_path = "paper_results/Llama-3.1-70B-Instruct/20250724_140425/final_results.json"
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found!")
        sys.exit(1)
    
    # Extract timestamp from file path or use current timestamp
    try:
        timestamp = file_path.split("/")[-2]
    except:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define and create output directory
    output_dir = f"images/subdata/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define log file path
    log_file_path = os.path.join(output_dir, "analysis_output.txt")

    # Redirect all print statements to the log file
    with open(log_file_path, 'w') as log_file:
        with redirect_stdout(log_file):
            # Load the data
            df = load_predictions(file_path)
    
            # Run the analysis
            results = analyze_persona_bias(df, output_dir)

    # This final print will go to the console because the 'with' block has ended
    print(f"\nAnalysis complete! Check the '{output_dir}' folder for visualizations and '{os.path.basename(log_file_path)}'.")