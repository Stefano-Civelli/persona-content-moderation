import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import simpledorff

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_process_data(file_path):
    """Load the results file and process into a DataFrame"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None, None
    
    # Extract predictions into DataFrame
    predictions = data["results"]
    df = pd.DataFrame(predictions)
    
    # Extract labels
    df['is_hate_speech'] = df['predicted_labels'].apply(lambda x: x['is_hate_speech'])
    df['target_category'] = df['predicted_labels'].apply(lambda x: x['target_category'])
    
    # Convert boolean to int for consistency
    df['is_hate_speech'] = df['is_hate_speech'].astype(int)
    
    return df, data

def compute_krippendorff_alpha(df, experiment_col, annotator_col, class_col):
    """Compute Krippendorff's Alpha for given columns"""
    try:
        alpha_score = simpledorff.calculate_krippendorffs_alpha_for_df(
            df,
            experiment_col=experiment_col,
            annotator_col=annotator_col,
            class_col=class_col
        )
        return alpha_score if not np.isnan(alpha_score) else 0.0
    except:
        return 0.0

def get_target_category_agreements(df, positions):
    """Compute agreement matrices for different target categories"""
    
    # Get unique target categories
    target_categories = df['target_category'].unique()
    
    results = {}
    
    for category in target_categories:
        if category == 'none':  # Skip 'none' category as it's not a specific target
            continue
            
        # Filter data for this target category
        category_df = df[df['target_category'] == category].copy()
        
        if len(category_df) < 10:  # Skip if too few samples
            continue
            
        # Create agreement matrix for this category
        matrix_size = len(positions)
        agreement_matrix = np.zeros((matrix_size, matrix_size))
        
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                if i == j:
                    # Intra-position agreement
                    pos_df = category_df[category_df['persona_pos'] == pos1]
                    if len(pos_df) > 0 and pos_df['persona_id'].nunique() >= 2:
                        alpha = compute_krippendorff_alpha(
                            pos_df, 'item_id', 'persona_id', 'is_hate_speech'
                        )
                        agreement_matrix[i, j] = alpha
                else:
                    # Inter-position agreement
                    combined_df = category_df[category_df['persona_pos'].isin([pos1, pos2])]
                    if len(combined_df) > 0 and combined_df['persona_id'].nunique() >= 2:
                        alpha = compute_krippendorff_alpha(
                            combined_df, 'item_id', 'persona_id', 'is_hate_speech'
                        )
                        agreement_matrix[i, j] = alpha
                        
        results[category] = agreement_matrix
    
    return results

def create_comprehensive_agreement_plot(df, positions, output_path):
    """Create a comprehensive plot showing agreement patterns"""
    
    # Get target category agreements
    target_agreements = get_target_category_agreements(df, positions)
    
    if not target_agreements:
        print("No target categories with sufficient data found.")
        return
    
    # Prepare data for visualization
    categories = list(target_agreements.keys())
    n_categories = len(categories)
    n_positions = len(positions)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Create a large subplot for the main heatmap
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], width_ratios=[1, 1, 1, 1])
    
    # Main heatmap showing aggregated intra vs inter diagonal patterns
    ax_main = fig.add_subplot(gs[0, :])
    
    # Compute intra-diagonal and inter-diagonal averages for each category
    intra_diag_scores = []
    inter_diag_scores = []
    category_names = []
    
    for category, matrix in target_agreements.items():
        # Intra-diagonal (same position)
        intra_scores = [matrix[i, i] for i in range(n_positions) if matrix[i, i] > 0]
        intra_avg = np.mean(intra_scores) if intra_scores else 0
        
        # Inter-diagonal (different positions)
        inter_scores = []
        for i in range(n_positions):
            for j in range(n_positions):
                if i != j and matrix[i, j] > 0:
                    inter_scores.append(matrix[i, j])
        inter_avg = np.mean(inter_scores) if inter_scores else 0
        
        intra_diag_scores.append(intra_avg)
        inter_diag_scores.append(inter_avg)
        category_names.append(category.upper())
    
    # Create comparison plot
    x = np.arange(len(category_names))
    width = 0.35
    
    bars1 = ax_main.bar(x - width/2, intra_diag_scores, width, label='Intra-Position Agreement', 
                       color='#2E86AB', alpha=0.8)
    bars2 = ax_main.bar(x + width/2, inter_diag_scores, width, label='Inter-Position Agreement', 
                       color='#A23B72', alpha=0.8)
    
    ax_main.set_xlabel('Target Categories', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Krippendorff\'s Alpha', fontsize=14, fontweight='bold')
    ax_main.set_title('Agreement Patterns by Target Category\n(Intra-Position vs Inter-Position)', 
                     fontsize=16, fontweight='bold', pad=20)
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(category_names, rotation=45, ha='right')
    ax_main.legend(fontsize=12)
    ax_main.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax_main.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax_main.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Position labels mapping
    pos_labels = {
        'top_left': 'TL', 'top_right': 'TR', 
        'bottom_left': 'BL', 'bottom_right': 'BR',
        'leftmost': 'L', 'rightmost': 'R'
    }
    
    # Individual category heatmaps (show top 4 categories)
    top_categories = sorted(target_agreements.keys())[:4]
    
    for idx, category in enumerate(top_categories):
        if idx >= 4:
            break
            
        ax = fig.add_subplot(gs[1 + idx//2, idx%2])
        matrix = target_agreements[category]
        
        # Create position labels
        pos_labels_list = [pos_labels.get(pos, pos) for pos in positions]
        
        # Plot heatmap
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        sns.heatmap(matrix, mask=mask, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   vmin=0, vmax=1, square=True, ax=ax,
                   xticklabels=pos_labels_list, yticklabels=pos_labels_list,
                   cbar_kws={'shrink': 0.8})
        
        ax.set_title(f'{category.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return intra_diag_scores, inter_diag_scores, category_names

def create_difference_analysis_plot(df, positions, output_path):
    """Create a plot showing the difference between intra and inter position agreements"""
    
    target_agreements = get_target_category_agreements(df, positions)
    
    if not target_agreements:
        return
    
    # Calculate differences
    differences = []
    category_names = []
    
    for category, matrix in target_agreements.items():
        n_positions = len(positions)
        
        # Intra-diagonal average
        intra_scores = [matrix[i, i] for i in range(n_positions) if matrix[i, i] > 0]
        intra_avg = np.mean(intra_scores) if intra_scores else 0
        
        # Inter-diagonal average
        inter_scores = []
        for i in range(n_positions):
            for j in range(n_positions):
                if i != j and matrix[i, j] > 0:
                    inter_scores.append(matrix[i, j])
        inter_avg = np.mean(inter_scores) if inter_scores else 0
        
        difference = intra_avg - inter_avg
        differences.append(difference)
        category_names.append(category.upper())
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color bars based on positive/negative difference
    colors = ['#2E86AB' if diff > 0 else '#A23B72' for diff in differences]
    
    bars = ax.bar(category_names, differences, color=colors, alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Target Categories', fontsize=14, fontweight='bold')
    ax.set_ylabel('Difference in Agreement\n(Intra - Inter)', fontsize=14, fontweight='bold')
    ax.set_title('Polarization Index by Target Category\n(Positive = More Intra-Position Agreement)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels
    for bar, diff in zip(bars, differences):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.005 if height > 0 else -0.015),
                f'{diff:.3f}', ha='center', va='bottom' if height > 0 else 'top', 
                fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    # Update this path to your actual results file
    file_path = "data/results/text_classification/Qwen2.5-32B-Instruct/20250704_183607.json"
    
    # Load and process data
    df, data = load_and_process_data(file_path)
    
    if df is None:
        print("Failed to load data")
        return
    
    # Determine positions (adapt this based on your actual data structure)
    positions = df['persona_pos'].unique().tolist()
    
    print(f"Dataset size: {len(df)} rows")
    print(f"Unique items: {df['item_id'].nunique()}")
    print(f"Positions present: {positions}")
    print(f"Target categories: {df['target_category'].unique()}")
    
    # Create comprehensive agreement plot
    output_path1 = "target_category_agreement_analysis.png"
    intra_scores, inter_scores, categories = create_comprehensive_agreement_plot(
        df, positions, output_path1
    )
    print(f"Comprehensive agreement plot saved to {output_path1}")
    
    # Create difference analysis plot
    output_path2 = "target_category_polarization_index.png"
    create_difference_analysis_plot(df, positions, output_path2)
    print(f"Polarization index plot saved to {output_path2}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    for cat, intra, inter in zip(categories, intra_scores, inter_scores):
        print(f"{cat:15}: Intra={intra:.3f}, Inter={inter:.3f}, Diff={intra-inter:.3f}")

if __name__ == "__main__":
    main()