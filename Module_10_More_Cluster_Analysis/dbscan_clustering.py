import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
import os

# Create output directory for figures
output_dir = "dbscan_figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set style for better plots - handle different versions
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')

# Try to import seaborn for color palette
try:
    import seaborn as sns
    sns.set_palette("husl")
except ImportError:
    pass

def generate_sample_data(dataset_type='blobs', n_samples=300, noise=0.1):
    """Generate different types of sample datasets for clustering"""
    if dataset_type == 'blobs':
        X, _ = make_blobs(n_samples=n_samples, centers=4, n_features=2, 
                         random_state=42, cluster_std=1.5)
    elif dataset_type == 'circles':
        X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.6, 
                           random_state=42)
    elif dataset_type == 'moons':
        X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    else:
        raise ValueError("dataset_type must be 'blobs', 'circles', or 'moons'")
    
    return X

def perform_dbscan_clustering(X, eps=0.5, min_samples=5):
    """Perform DBSCAN clustering and identify point types"""
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_scaled)
    
    # Identify core samples
    core_samples_mask = np.zeros_like(cluster_labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    
    # Classify points
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    # Create point type classification
    point_types = np.full(len(X), 'border', dtype=object)
    point_types[core_samples_mask] = 'core'
    point_types[cluster_labels == -1] = 'noise'
    
    return cluster_labels, point_types, n_clusters, n_noise, X_scaled

def plot_dbscan_results(X, cluster_labels, point_types, title, eps, min_samples, save_name=None):
    """Plot DBSCAN clustering results with color-coded point types"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Define colors for point types
    colors = {'core': 'red', 'border': 'yellow', 'noise': 'blue'}
    
    # Plot each point type
    for point_type in ['noise', 'border', 'core']:  # Plot noise first, core last
        mask = point_types == point_type
        if np.any(mask):
            scatter = ax.scatter(X[mask, 0], X[mask, 1], 
                               c=colors[point_type], 
                               s=60 if point_type == 'core' else 30,
                               alpha=0.8 if point_type != 'noise' else 0.6,
                               edgecolors='black' if point_type == 'core' else 'none',
                               linewidth=0.5,
                               label=f'{point_type.capitalize()} points')
    
    # Count statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    n_core = np.sum(point_types == 'core')
    n_border = np.sum(point_types == 'border')
    
    ax.set_title(f'{title}\nClusters: {n_clusters}, Core: {n_core}, Border: {n_border}, Noise: {n_noise}\n'
                f'eps={eps}, min_samples={min_samples}', fontsize=12, pad=20)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if save_name is provided
    if save_name:
        save_path = os.path.join(output_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved: {save_path}")
    
    return fig

def compare_parameters(X, eps_values, min_samples_values, save_name=None):
    """Compare DBSCAN results with different parameter combinations"""
    fig, axes = plt.subplots(len(eps_values), len(min_samples_values), 
                            figsize=(5*len(min_samples_values), 4*len(eps_values)))
    
    if len(eps_values) == 1:
        axes = axes.reshape(1, -1)
    if len(min_samples_values) == 1:
        axes = axes.reshape(-1, 1)
    
    colors = {'core': 'red', 'border': 'yellow', 'noise': 'blue'}
    
    for i, eps in enumerate(eps_values):
        for j, min_samples in enumerate(min_samples_values):
            ax = axes[i, j]
            
            # Perform clustering
            cluster_labels, point_types, n_clusters, n_noise, X_scaled = perform_dbscan_clustering(
                X, eps=eps, min_samples=min_samples)
            
            # Plot results
            for point_type in ['noise', 'border', 'core']:
                mask = point_types == point_type
                if np.any(mask):
                    ax.scatter(X[mask, 0], X[mask, 1], 
                             c=colors[point_type], 
                             s=40 if point_type == 'core' else 20,
                             alpha=0.8 if point_type != 'noise' else 0.6,
                             edgecolors='black' if point_type == 'core' else 'none',
                             linewidth=0.3)
            
            ax.set_title(f'eps={eps}, min_samples={min_samples}\n'
                        f'Clusters: {n_clusters}, Noise: {n_noise}', fontsize=10)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if save_name is provided
    if save_name:
        save_path = os.path.join(output_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved: {save_path}")
    
    return fig

def main():
    """Main function to demonstrate DBSCAN clustering"""
    print("DBSCAN Clustering Demo")
    print("=" * 50)
    
    # Generate different datasets
    datasets = {
        'Blob Data': generate_sample_data('blobs', n_samples=300),
        'Circle Data': generate_sample_data('circles', n_samples=300, noise=0.1),
        'Moon Data': generate_sample_data('moons', n_samples=300, noise=0.1)
    }
    
    # Optimal parameters for each dataset type
    optimal_params = {
        'Blob Data': {'eps': 0.5, 'min_samples': 5},
        'Circle Data': {'eps': 0.3, 'min_samples': 5},
        'Moon Data': {'eps': 0.3, 'min_samples': 5}
    }
    
    # Plot results for each dataset
    for dataset_name, X in datasets.items():
        print(f"\nProcessing {dataset_name}...")
        
        params = optimal_params[dataset_name]
        cluster_labels, point_types, n_clusters, n_noise, X_scaled = perform_dbscan_clustering(
            X, eps=params['eps'], min_samples=params['min_samples'])
        
        # Print statistics
        n_core = np.sum(point_types == 'core')
        n_border = np.sum(point_types == 'border')
        
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Core points: {n_core}")
        print(f"  Border points: {n_border}")
        print(f"  Noise points: {n_noise}")
        
        # Create visualization and save
        save_name = f"dbscan_{dataset_name.lower().replace(' ', '_')}"
        fig = plot_dbscan_results(X, cluster_labels, point_types, 
                                 f'DBSCAN Clustering - {dataset_name}',
                                 params['eps'], params['min_samples'], save_name)
        plt.show()
    
    # Parameter comparison for blob data
    print("\nParameter Comparison for Blob Data...")
    X_blobs = datasets['Blob Data']
    
    eps_values = [0.3, 0.5, 0.8]
    min_samples_values = [3, 5, 10]
    
    fig = compare_parameters(X_blobs, eps_values, min_samples_values, "parameter_comparison")
    plt.suptitle('DBSCAN Parameter Comparison\nRed=Core, Yellow=Border, Blue=Noise', 
                 fontsize=14, y=0.98)
    plt.show()
    
    # Demonstrate the effect of different eps values
    print("\nEps Parameter Analysis...")
    X_circles = datasets['Circle Data']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    eps_test_values = [0.1, 0.2, 0.3, 0.5]
    colors = {'core': 'red', 'border': 'yellow', 'noise': 'blue'}
    
    for i, eps in enumerate(eps_test_values):
        cluster_labels, point_types, n_clusters, n_noise, _ = perform_dbscan_clustering(
            X_circles, eps=eps, min_samples=5)
        
        for point_type in ['noise', 'border', 'core']:
            mask = point_types == point_type
            if np.any(mask):
                axes[i].scatter(X_circles[mask, 0], X_circles[mask, 1], 
                               c=colors[point_type], 
                               s=50 if point_type == 'core' else 30,
                               alpha=0.8 if point_type != 'noise' else 0.6,
                               edgecolors='black' if point_type == 'core' else 'none',
                               linewidth=0.5)
        
        axes[i].set_title(f'eps = {eps}\nClusters: {n_clusters}, Noise: {n_noise}')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Effect of eps Parameter on Circle Data\nRed=Core, Yellow=Border, Blue=Noise', 
                 fontsize=14)
    plt.tight_layout()
    
    # Save the eps analysis figure
    save_path = os.path.join(output_dir, "eps_parameter_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Figure saved: {save_path}")
    plt.show()
    
    print("\nDBSCAN Clustering Demo Complete!")
    print(f"\nAll figures saved in the '{output_dir}' directory:")
    print("  - dbscan_blob_data.png")
    print("  - dbscan_circle_data.png") 
    print("  - dbscan_moon_data.png")
    print("  - parameter_comparison.png")
    print("  - eps_parameter_analysis.png")
    print("\nColor Legend:")
    print("  Red = Core points (dense regions)")
    print("  Yellow = Border points (edge of clusters)")
    print("  Blue = Noise points (outliers)")

if __name__ == "__main__":
    main()