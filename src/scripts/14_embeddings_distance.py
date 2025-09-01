import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

def load_persona_embeddings(file_path):
    """Load persona embeddings from pickle file."""
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    return np.array(embeddings)

def load_behavioral_embeddings(file_path, model_name=None, run_id=None):
    """
    Load behavioral embeddings from CSV file.
    
    Parameters:
    - file_path: path to CSV file
    - model_name: filter by specific model (optional)
    - run_id: filter by specific run (optional)
    
    Returns:
    - behavioral_embeddings: array of UMAP coordinates
    - persona_ids: array of corresponding persona IDs
    """
    df = pd.read_csv(file_path)
    
    # Filter by model and run if specified
    if model_name:
        df = df[df['model_name'] == model_name]
    if run_id is not None:
        df = df[df['run_id'] == run_id]
    
    if len(df) == 0:
        raise ValueError(f"No data found for model_name='{model_name}', run_id={run_id}")
    
    # Sort by persona_id to ensure consistent ordering
    df = df.sort_values('persona_id')
    
    behavioral_embeddings = df[['umap1', 'umap2']].values
    persona_ids = df['persona_id'].values
    
    return behavioral_embeddings, persona_ids

def compute_pairwise_distances(embeddings, metric='euclidean'):
    """Compute pairwise distances between all embeddings."""
    distances = pdist(embeddings, metric=metric)
    return distances

def plot_distance_correlation(persona_distances, behavioral_distances, 
                            title="Persona vs Behavioral Distance Correlation",
                            sample_size=None):
    """
    Plot scatter plot of distances from two embedding spaces.
    
    Parameters:
    - persona_distances: 1D array of pairwise distances from persona embeddings
    - behavioral_distances: 1D array of pairwise distances from behavioral embeddings  
    - title: Plot title
    - sample_size: If provided, randomly sample this many points for plotting
    """
    
    # Optionally sample points for cleaner visualization
    if sample_size and len(persona_distances) > sample_size:
        indices = np.random.choice(len(persona_distances), sample_size, replace=False)
        persona_dist_sample = persona_distances[indices]
        behavioral_dist_sample = behavioral_distances[indices]
    else:
        persona_dist_sample = persona_distances
        behavioral_dist_sample = behavioral_distances
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(persona_dist_sample, behavioral_dist_sample, alpha=0.6, s=20)
    
    # Add regression line
    z = np.polyfit(persona_dist_sample, behavioral_dist_sample, 1)
    p = np.poly1d(z)
    plt.plot(persona_dist_sample, p(persona_dist_sample), "r--", alpha=0.8, linewidth=2)
    
    # Calculate correlations
    pearson_r, pearson_p = pearsonr(persona_distances, behavioral_distances)
    spearman_r, spearman_p = spearmanr(persona_distances, behavioral_distances)
    
    # Add labels and title
    plt.xlabel('Persona Embedding Distances', fontsize=12)
    plt.ylabel('Behavioral Embedding Distances', fontsize=12) 
    plt.title(f'{title}\nPearson r = {pearson_r:.3f} (p = {pearson_p:.3e})\n' +
              f'Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.3e})', fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return pearson_r, spearman_r

def main():
    # File paths (replace with your actual paths)
    persona_embeddings_path = "data/interim/embeddings_all-MiniLM-L6-v2_200000docs_636d9a81.pkl"
    behavioral_embeddings_path = "data/results/behavioural_embeddings/Qwen2.5-32B-Instruct/20250717_215350/embedding_data_n5_dist0.1_run1.csv"
    
    # Specify model and run to analyze
    model_name = "Qwen2.5-32B-Instruct"  # Change this to your desired model
    run_id = 1  # Change this to your desired run
    
    print("Loading embeddings...")
    
    # Load persona embeddings (assumes ordered by persona ID starting from 1)
    persona_embeddings_all = load_persona_embeddings(persona_embeddings_path)
    print(f"Total persona embeddings loaded: {len(persona_embeddings_all)}")
    
    # Load behavioral embeddings for specific model/run
    behavioral_embeddings, persona_ids = load_behavioral_embeddings(
        behavioral_embeddings_path, model_name=model_name, run_id=run_id
    )
    print(f"Behavioral embeddings shape: {behavioral_embeddings.shape}")
    print(f"Persona IDs range: {persona_ids.min()} to {persona_ids.max()}")
    
    # Extract corresponding persona embeddings
    # Convert persona IDs to 0-based indices (assuming persona IDs start from 1)
    persona_indices = persona_ids - 1
    
    # Validate indices are within bounds
    if persona_indices.max() >= len(persona_embeddings_all):
        raise ValueError(f"Persona ID {persona_ids.max()} exceeds available embeddings ({len(persona_embeddings_all)})")
    
    persona_embeddings = persona_embeddings_all[persona_indices]
    
    print(f"Matched persona embeddings shape: {persona_embeddings.shape}")
    print(f"Number of personas to analyze: {len(persona_embeddings)}")
    
    print("Computing pairwise distances...")
    
    # Compute pairwise distances
    persona_distances = compute_pairwise_distances(persona_embeddings, metric='cosine')
    behavioral_distances = compute_pairwise_distances(behavioral_embeddings, metric='euclidean')
    
    print(f"Number of pairwise distances: {len(persona_distances)}")
    print(f"Persona distances - min: {persona_distances.min():.3f}, max: {persona_distances.max():.3f}")
    print(f"Behavioral distances - min: {behavioral_distances.min():.3f}, max: {behavioral_distances.max():.3f}")
    
    # Create correlation plot
    print("Creating correlation plot...")
    pearson_r, spearman_r = plot_distance_correlation(
        persona_distances, 
        behavioral_distances,
        title=f"Persona vs Behavioral Distance Correlation\n{model_name} (Run {run_id})",
        sample_size=10000  # Sample for cleaner visualization if too many points
    )
    
    # Print correlation statistics
    print(f"\nCorrelation Results:")
    print(f"Pearson correlation: {pearson_r:.4f}")
    print(f"Spearman correlation: {spearman_r:.4f}")
    
    # Save plot
    output_filename = f'persona_behavioral_distance_correlation_{model_name.replace(".", "_")}_run{run_id}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{output_filename}'")
    
    plt.show()

if __name__ == "__main__":
    main()