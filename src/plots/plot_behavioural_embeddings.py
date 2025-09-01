import argparse
import warnings
from pathlib import Path
from multiprocessing import Pool
import itertools
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from sklearn.metrics import silhouette_score
from matplotlib.patches import Ellipse

# Suppress warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PLOT STYLING PARAMETERS - MODIFY THESE TO CUSTOMIZE YOUR PLOTS
# =============================================================================

# Font sizes
AXIS_LABEL_SIZE = 24
LEGEND_FONT_SIZE = 18
TICK_LABEL_SIZE = 11

# Figure dimensions
FIGURE_WIDTH = 10
FIGURE_HEIGHT = 8

# Point styling
POINT_SIZE = 120
POINT_ALPHA = 0.8
POINT_EDGE_COLOR = "black"
POINT_EDGE_WIDTH = 0.5

# Legend styling
LEGEND_POSITION = "upper right"  # Options: 'upper right', 'upper left', 'lower right', 'lower left', 'center', etc.
LEGEND_FRAMEON = True
LEGEND_FRAME_ALPHA = 0.9
LEGEND_SHADOW = False

# Axis styling
SHOW_AXIS_LABELS = True  # Set to False to hide axis labels completely
SHOW_TICK_LABELS = False  # Set to False to hide tick labels (numbers)
SHOW_GRID = True
GRID_ALPHA = 0.3

# Clustering quality visualization options
SHOW_CLUSTER_BOUNDARIES = False  # Show confidence ellipses around clusters
SHOW_DENSITY_CONTOURS = False   # Show density contours (can be cluttered with ellipses)
SHOW_SILHOUETTE_SCORE = True    # Display silhouette score as text annotation

# Persona ID label options
SHOW_PERSONA_IDS = True         # Show persona IDs as text labels on points
PERSONA_ID_FONT_SIZE = 8         # Font size for persona ID labels
PERSONA_ID_OFFSET_X = 0.02       # Horizontal offset for ID labels (relative to data range)
PERSONA_ID_OFFSET_Y = 0.02       # Vertical offset for ID labels (relative to data range)
PERSONA_ID_COLOR = "black"       # Color for persona ID text
PERSONA_ID_ALPHA = 0.8           # Transparency for persona ID text
PERSONA_ID_WEIGHT = "bold"       # Font weight: 'normal', 'bold', etc.
PERSONA_ID_BACKGROUND = True     # Add white background to ID labels for better readability
PERSONA_ID_BACKGROUND_ALPHA = 0.7  # Transparency of the background box

# Boundary styling
BOUNDARY_ALPHA = 0.2
BOUNDARY_LINE_WIDTH = 2
BOUNDARY_CONFIDENCE = 0.95  # Confidence level for ellipses (0.68 = 1σ, 0.95 = 2σ)

# Color palette
CORNER_COLORS = {
    "top_left": "#2E86AB",      # Professional blue
    "top_right": "#A23B72",     # Professional magenta
    "bottom_left": "#F18F01",   # Professional orange
    "bottom_right": "#C73E1D",  # Professional red
    "leftmost": "#2E86AB",      # Professional blue
    "rightmost": "#C73E1D"      # Professional red
}

# Plot style
PLOT_STYLE = "seaborn-v0_8-whitegrid"  # Options: 'seaborn-v0_8-whitegrid', 'seaborn-v0_8-white', 'classic', etc.

# =============================================================================
# END OF CUSTOMIZATION PARAMETERS
# =============================================================================


def calculate_confidence_ellipse(x, y, confidence=0.95):
    """Calculate confidence ellipse parameters for a cluster."""
    if len(x) < 3:  # Need at least 3 points for meaningful ellipse
        return None
    
    # Calculate covariance matrix
    data = np.column_stack([x, y])
    cov = np.cov(data.T)
    
    # Calculate eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Calculate ellipse parameters
    # Chi-squared value for confidence level
    if confidence == 0.68:
        chi2_val = 2.279  # 1 sigma
    elif confidence == 0.95:
        chi2_val = 5.991  # 2 sigma
    else:
        from scipy.stats import chi2
        chi2_val = chi2.ppf(confidence, df=2)
    
    # Width and height of ellipse
    width = 2 * np.sqrt(chi2_val * eigenvals[0])
    height = 2 * np.sqrt(chi2_val * eigenvals[1])
    
    # Angle of ellipse
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    
    # Center of ellipse
    center_x = np.mean(x)
    center_y = np.mean(y)
    
    return {
        'center': (center_x, center_y),
        'width': width,
        'height': height,
        'angle': angle
    }


def add_density_contours(ax, x, y, color, levels=3, alpha=0.3, linewidth=1):
    """Add density contours for a cluster."""
    if len(x) < 5:  # Need sufficient points for density estimation
        return
    
    try:
        # Create density estimation
        xy = np.column_stack([x, y])
        kde = gaussian_kde(xy.T)
        
        # Create grid for contours
        x_min, x_max = x.min() - 0.5, x.max() + 0.5
        y_min, y_max = y.min() - 0.5, y.max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                           np.linspace(y_min, y_max, 50))
        
        # Calculate density
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        density = kde(positions.T).reshape(xx.shape)
        
        # Add contours
        ax.contour(xx, yy, density, levels=levels, colors=[color], 
                  alpha=alpha, linewidths=linewidth)
    except:
        # Skip if density estimation fails
        pass


def add_persona_id_labels(ax, persona_info):
    """Add persona ID labels to scatter plot points."""
    if not SHOW_PERSONA_IDS:
        return
    
    # Calculate data range for offset scaling
    x_range = persona_info['umap1'].max() - persona_info['umap1'].min()
    y_range = persona_info['umap2'].max() - persona_info['umap2'].min()
    
    # Calculate actual offsets
    x_offset = x_range * PERSONA_ID_OFFSET_X
    y_offset = y_range * PERSONA_ID_OFFSET_Y
    
    # Add text labels for each point
    for _, row in persona_info.iterrows():
        x = row['umap1'] + x_offset
        y = row['umap2'] + y_offset
        
        # Get persona ID (assuming there's a column for it)
        # Try different possible column names for persona ID
        persona_id = None
        for col_name in ['persona_id', 'id', 'persona_ID', 'ID', 'persona', 'index']:
            if col_name in row:
                persona_id = str(row[col_name])
                break
        
        # If no explicit ID column, use the DataFrame index
        if persona_id is None:
            persona_id = str(row.name)
        
        # Create text annotation
        text_props = {
            'fontsize': PERSONA_ID_FONT_SIZE,
            'color': PERSONA_ID_COLOR,
            'alpha': PERSONA_ID_ALPHA,
            'weight': PERSONA_ID_WEIGHT,
            'ha': 'center',
            'va': 'center'
        }
        
        # Add background box if requested
        if PERSONA_ID_BACKGROUND:
            bbox_props = dict(
                boxstyle="round,pad=0.2",
                facecolor='white',
                alpha=PERSONA_ID_BACKGROUND_ALPHA,
                edgecolor='none'
            )
            text_props['bbox'] = bbox_props
        
        ax.text(x, y, persona_id, **text_props)


def create_plot(args_tuple):
    csv_path, base_output_path = args_tuple
    
    # Load the embedding data
    persona_info = pd.read_csv(csv_path)
    
    # Extract parameters from the data
    n_neighbors = persona_info['n_neighbors'].iloc[0]
    min_dist = persona_info['min_dist'].iloc[0]
    run_id = persona_info['run_id'].iloc[0]
    model_name = persona_info['model_name'].iloc[0]
    
    print(f"Creating plot: n_neighbors={n_neighbors}, min_dist={min_dist}, run={run_id}")
    
    # Set up the plot style
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    # Create the scatter plot
    sns.scatterplot(
        data=persona_info,
        x="umap1",
        y="umap2",
        hue="persona_pos",
        palette=CORNER_COLORS,
        s=POINT_SIZE,
        ax=ax,
        edgecolor=POINT_EDGE_COLOR,
        linewidth=POINT_EDGE_WIDTH,
        alpha=POINT_ALPHA,
    )

    # Add persona ID labels
    add_persona_id_labels(ax, persona_info)

    # Add clustering quality visualizations
    if SHOW_CLUSTER_BOUNDARIES or SHOW_DENSITY_CONTOURS:
        for position, color in CORNER_COLORS.items():
            cluster_data = persona_info[persona_info['persona_pos'] == position]
            if len(cluster_data) == 0:
                continue
                
            x = cluster_data['umap1'].values
            y = cluster_data['umap2'].values
            
            # Add confidence ellipses
            if SHOW_CLUSTER_BOUNDARIES:
                ellipse_params = calculate_confidence_ellipse(x, y, BOUNDARY_CONFIDENCE)
                if ellipse_params:
                    ellipse = Ellipse(
                        ellipse_params['center'],
                        ellipse_params['width'],
                        ellipse_params['height'],
                        angle=ellipse_params['angle'],
                        facecolor=color,
                        alpha=BOUNDARY_ALPHA,
                        edgecolor=color,
                        linewidth=BOUNDARY_LINE_WIDTH
                    )
                    ax.add_patch(ellipse)
            
            # Add density contours
            if SHOW_DENSITY_CONTOURS:
                add_density_contours(ax, x, y, color)

    # Calculate and display silhouette score
    if SHOW_SILHOUETTE_SCORE and len(persona_info) > 1:
        try:
            # Convert categorical labels to numeric
            label_map = {label: i for i, label in enumerate(persona_info['persona_pos'].unique())}
            numeric_labels = persona_info['persona_pos'].map(label_map)
            
            # Calculate silhouette score
            coordinates = persona_info[['umap1', 'umap2']].values
            sil_score = silhouette_score(coordinates, numeric_labels)
            
            # Add text annotation
            ax.text(0.02, 0.98, f'Silhouette Score: {sil_score:.3f}', 
                   transform=ax.transAxes, fontsize=LEGEND_FONT_SIZE-1,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8))
        except:
            # Skip if silhouette calculation fails
            pass

    # Customize axis labels
    if SHOW_AXIS_LABELS:
        ax.set_xlabel("UMAP Dimension 1", fontsize=AXIS_LABEL_SIZE, fontweight='bold')
        ax.set_ylabel("UMAP Dimension 2", fontsize=AXIS_LABEL_SIZE, fontweight='bold')
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Customize tick labels
    if not SHOW_TICK_LABELS:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
    else:
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)

    # Customize grid
    if SHOW_GRID:
        ax.grid(True, alpha=GRID_ALPHA)
    else:
        ax.grid(False)

    # Customize legend
    legend = ax.legend(
        title="Political Position",
        loc=LEGEND_POSITION,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
        frameon=LEGEND_FRAMEON,
        framealpha=LEGEND_FRAME_ALPHA,
        shadow=LEGEND_SHADOW,
        fancybox=True
    )
    
    # Make legend title bold
    legend.get_title().set_fontweight('bold')

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make left and bottom spines slightly thicker
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    plt.tight_layout()
    
    # Create output filename
    output_filename = f"persona_clusters_n{n_neighbors}_dist{min_dist}_run{run_id}.png"
    output_path = base_output_path / output_filename
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    
    print(f"Saved plot to: {output_path}")
    return output_path


def main():
    """Main function to parse arguments and generate plots from embedding data."""
    parser = argparse.ArgumentParser(
        description="Generate professional plots from saved persona embedding data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task_type", type=str, default="img", help="Type of task"
    )  # text, img
    parser.add_argument(
        "--timestamp",
        type=str,
        default="20250713_103643",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen2.5-32B-Instruct",
        help="Model name/path",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=6,
        help="Number of processes to use for multiprocessing",
    )
    parser.add_argument(
        "--input_pattern",
        type=str,
        default="embedding_data_*.csv",
        help="Pattern to match input CSV files",
    )

    args = parser.parse_args()

    MODEL_NAME = args.model.split("/")[-1]

    # Input and output paths
    input_base_path = Path(
        f"data/results/behavioural_embeddings/{MODEL_NAME}/{args.timestamp}"
    )
    output_base_path = Path(
        f"images/behavioural_embeddings/{MODEL_NAME}/{args.timestamp}"
    )

    # Find all CSV files matching the pattern
    csv_files = list(input_base_path.glob(args.input_pattern))
    
    if not csv_files:
        print(f"No CSV files found matching pattern '{args.input_pattern}' in {input_base_path}")
        return
    
    print(f"Found {len(csv_files)} embedding data files:")
    for csv_file in sorted(csv_files):
        print(f"  {csv_file.name}")
    print()

    # --- Prepare arguments for multiprocessing ---
    plot_args = [
        (csv_file, output_base_path)
        for csv_file in csv_files
    ]

    # --- Generate plots using multiprocessing ---
    print("="*70)
    print("GENERATING PROFESSIONAL PLOTS")
    print("="*70)
    
    with Pool(processes=args.num_processes) as pool:
        results = pool.map(create_plot, plot_args)
    
    print("="*70)
    print("ALL PLOTS GENERATED")
    print("="*70)
    print("Generated plots:")
    for result in results:
        print(f"  {result}")
    print(f"\nTotal plots generated: {len(results)}")


if __name__ == "__main__":
    main()