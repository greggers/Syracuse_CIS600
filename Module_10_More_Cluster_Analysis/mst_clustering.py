import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.cluster.hierarchy import fcluster, linkage
import networkx as nx
import os

# Create output directory for figures
output_dir = "mst_clustering_figures"
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

def generate_sample_data(dataset_type='blobs', n_samples=200, noise=0.1):
    """Generate different types of sample datasets for clustering"""
    if dataset_type == 'blobs':
        X, _ = make_blobs(n_samples=n_samples, centers=4, n_features=2, 
                         random_state=42, cluster_std=1.2)
    elif dataset_type == 'circles':
        X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.6, 
                           random_state=42)
    elif dataset_type == 'moons':
        X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset_type == 'spiral':
        # Create a simple spiral dataset
        t = np.linspace(0, 4*np.pi, n_samples)
        x = t * np.cos(t) + np.random.normal(0, 0.5, n_samples)
        y = t * np.sin(t) + np.random.normal(0, 0.5, n_samples)
        X = np.column_stack([x, y])
    else:
        raise ValueError("dataset_type must be 'blobs', 'circles', 'moons', or 'spiral'")
    
    return X

def compute_mst(X):
    """Compute the Minimum Spanning Tree from data points"""
    # Compute pairwise distances
    distances = pdist(X, metric='euclidean')
    distance_matrix = squareform(distances)
    
    # Compute MST using scipy
    mst = minimum_spanning_tree(distance_matrix).toarray()
    
    # Convert to edge list format
    edges = []
    edge_weights = []
    
    for i in range(len(mst)):
        for j in range(i+1, len(mst)):
            if mst[i, j] > 0:
                edges.append((i, j))
                edge_weights.append(mst[i, j])
            elif mst[j, i] > 0:
                edges.append((i, j))
                edge_weights.append(mst[j, i])
    
    return edges, edge_weights, mst

def mst_clustering(X, n_clusters=3, method='remove_longest'):
    """Perform clustering using MST by removing longest edges"""
    edges, edge_weights, mst_matrix = compute_mst(X)
    
    if method == 'remove_longest':
        # Sort edges by weight (descending) and remove the longest ones
        sorted_indices = np.argsort(edge_weights)[::-1]
        edges_to_remove = sorted_indices[:n_clusters-1]
        
        # Create adjacency matrix without the longest edges
        adj_matrix = mst_matrix.copy()
        for idx in edges_to_remove:
            i, j = edges[idx]
            adj_matrix[i, j] = 0
            adj_matrix[j, i] = 0
        
        # Find connected components
        labels = find_connected_components(adj_matrix)
        removed_edges = [edges[i] for i in edges_to_remove]
        
    elif method == 'hierarchical':
        # Use hierarchical clustering on MST distances
        # Create linkage matrix from MST
        distances = pdist(X, metric='euclidean')
        linkage_matrix = linkage(distances, method='single')
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
        removed_edges = []
    
    return labels, edges, edge_weights, removed_edges

def find_connected_components(adj_matrix):
    """Find connected components in adjacency matrix using DFS"""
    n = len(adj_matrix)
    visited = np.zeros(n, dtype=bool)
    labels = np.zeros(n, dtype=int)
    current_label = 0
    
    def dfs(node, label):
        visited[node] = True
        labels[node] = label
        for neighbor in range(n):
            if adj_matrix[node, neighbor] > 0 and not visited[neighbor]:
                dfs(neighbor, label)
    
    for i in range(n):
        if not visited[i]:
            dfs(i, current_label)
            current_label += 1
    
    return labels

def plot_mst_clustering(X, labels, edges, edge_weights, removed_edges, title, save_name=None):
    """Plot MST clustering results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Define colors for clusters
    colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(labels))))
    
    # Plot 1: Original data with MST
    ax1.scatter(X[:, 0], X[:, 1], c='lightblue', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Draw MST edges
    for i, (start, end) in enumerate(edges):
        x_coords = [X[start, 0], X[end, 0]]
        y_coords = [X[start, 1], X[end, 1]]
        
        # Color removed edges differently
        if (start, end) in removed_edges or (end, start) in removed_edges:
            ax1.plot(x_coords, y_coords, 'r-', linewidth=3, alpha=0.8, label='Removed edges' if i == 0 else "")
        else:
            ax1.plot(x_coords, y_coords, 'k-', linewidth=1, alpha=0.6)
    
    ax1.set_title(f'{title}\nMinimum Spanning Tree', fontsize=12)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.grid(True, alpha=0.3)
    if removed_edges:
        ax1.legend()
    
    # Plot 2: Clustered data
    for i, color in enumerate(colors):
        mask = labels == i
        if np.any(mask):
            ax2.scatter(X[mask, 0], X[mask, 1], c=[color], s=60, 
                       alpha=0.8, edgecolors='black', linewidth=0.5,
                       label=f'Cluster {i+1}')
    
    # Draw remaining MST edges (after clustering)
    for start, end in edges:
        if (start, end) not in removed_edges and (end, start) not in removed_edges:
            if labels[start] == labels[end]:  # Only draw edges within same cluster
                x_coords = [X[start, 0], X[end, 0]]
                y_coords = [X[start, 1], X[end, 1]]
                ax2.plot(x_coords, y_coords, 'gray', linewidth=1, alpha=0.4)
    
    n_clusters = len(np.unique(labels))
    ax2.set_title(f'MST Clustering Result\nClusters: {n_clusters}', fontsize=12)
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if save_name is provided
    if save_name:
        save_path = os.path.join(output_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved: {save_path}")
    
    return fig

def plot_mst_edge_weights(edge_weights, n_clusters, title, save_name=None):
    """Plot histogram of MST edge weights to show clustering threshold"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Sort edge weights
    sorted_weights = np.sort(edge_weights)
    
    # Plot histogram
    ax.hist(edge_weights, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Mark the threshold (longest edges that were removed)
    if n_clusters > 1:
        threshold_idx = len(sorted_weights) - (n_clusters - 1)
        threshold = sorted_weights[threshold_idx]
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                  label=f'Clustering threshold: {threshold:.3f}')
        ax.legend()
    
    ax.set_title(f'{title}\nDistribution of MST Edge Weights', fontsize=12)
    ax.set_xlabel('Edge Weight (Distance)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if save_name is provided
    if save_name:
        save_path = os.path.join(output_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved: {save_path}")
    
    return fig

def compare_cluster_numbers(X, dataset_name, max_clusters=6):
    """Compare MST clustering with different numbers of clusters"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = plt.cm.Set3(np.linspace(0, 1, max_clusters))
    
    for k in range(2, max_clusters + 2):
        ax = axes[k-2]
        
        # Perform clustering
        labels, edges, edge_weights, removed_edges = mst_clustering(X, n_clusters=k)
        
        # Plot results
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                ax.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], s=50, 
                          alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Draw remaining MST edges
        for start, end in edges:
            if (start, end) not in removed_edges and (end, start) not in removed_edges:
                if labels[start] == labels[end]:
                    x_coords = [X[start, 0], X[end, 0]]
                    y_coords = [X[start, 1], X[end, 1]]
                    ax.plot(x_coords, y_coords, 'gray', linewidth=0.8, alpha=0.4)
        
        ax.set_title(f'k = {k} clusters', fontsize=11)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'MST Clustering Comparison - {dataset_name}', fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save figure
    save_name = f"mst_comparison_{dataset_name.lower().replace(' ', '_')}"
    save_path = os.path.join(output_dir, f"{save_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Figure saved: {save_path}")
    
    return fig

def main():
    """Main function to demonstrate MST clustering"""
    print("Minimum Spanning Tree (MST) Clustering Demo")
    print("=" * 60)
    
    # Generate different datasets
    datasets = {
        'Blob Data': generate_sample_data('blobs', n_samples=150),
        'Circle Data': generate_sample_data('circles', n_samples=150, noise=0.08),
        'Moon Data': generate_sample_data('moons', n_samples=150, noise=0.08),
        'Spiral Data': generate_sample_data('spiral', n_samples=100)
    }
    
    # Optimal number of clusters for each dataset
    optimal_clusters = {
        'Blob Data': 4,
        'Circle Data': 2,
        'Moon Data': 2,
        'Spiral Data': 3
    }
    
    # Process each dataset
    for dataset_name, X in datasets.items():
        print(f"\nProcessing {dataset_name}...")
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_clusters = optimal_clusters[dataset_name]
        
        # Perform MST clustering
        labels, edges, edge_weights, removed_edges = mst_clustering(X_scaled, n_clusters=n_clusters)
        
        # Print statistics
        unique_labels = np.unique(labels)
        print(f"  Number of clusters found: {len(unique_labels)}")
        for i in unique_labels:
            count = np.sum(labels == i)
            print(f"  Cluster {i+1}: {count} points")
        
        # Plot MST clustering results
        save_name = f"mst_{dataset_name.lower().replace(' ', '_')}"
        fig1 = plot_mst_clustering(X_scaled, labels, edges, edge_weights, removed_edges,
                                  f'MST Clustering - {dataset_name}', save_name)
        plt.show()
        
        # Plot edge weight distribution
        save_name_weights = f"mst_weights_{dataset_name.lower().replace(' ', '_')}"
        fig2 = plot_mst_edge_weights(edge_weights, n_clusters, 
                                    f'MST Edge Weights - {dataset_name}', save_name_weights)
        plt.show()
    
    # Compare different numbers of clusters for blob data
    print("\nCluster Number Comparison for Blob Data...")
    X_blobs = datasets['Blob Data']
    scaler = StandardScaler()
    X_blobs_scaled = scaler.fit_transform(X_blobs)
    
    fig3 = compare_cluster_numbers(X_blobs_scaled, 'Blob Data')
    plt.show()
    
    # Demonstrate MST properties
    print("\nMST Properties Analysis...")
    X_demo = datasets['Blob Data']
    scaler = StandardScaler()
    X_demo_scaled = scaler.fit_transform(X_demo)
    
    edges, edge_weights, mst_matrix = compute_mst(X_demo_scaled)
    
    # Create a detailed MST analysis plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Original data points
    ax1.scatter(X_demo_scaled[:, 0], X_demo_scaled[:, 1], c='lightblue', s=60, 
               alpha=0.8, edgecolors='black', linewidth=0.5)
    ax1.set_title('Original Data Points', fontsize=12)
    ax1.set_xlabel('Feature 1 (scaled)')
    ax1.set_ylabel('Feature 2 (scaled)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Complete MST
    ax2.scatter(X_demo_scaled[:, 0], X_demo_scaled[:, 1], c='lightblue', s=60, 
               alpha=0.8, edgecolors='black', linewidth=0.5)
    for start, end in edges:
        x_coords = [X_demo_scaled[start, 0], X_demo_scaled[end, 0]]
        y_coords = [X_demo_scaled[start, 1], X_demo_scaled[end, 1]]
        ax2.plot(x_coords, y_coords, 'k-', linewidth=1, alpha=0.7)
    ax2.set_title(f'Complete MST ({len(edges)} edges)', fontsize=12)
    ax2.set_xlabel('Feature 1 (scaled)')
    ax2.set_ylabel('Feature 2 (scaled)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: MST with longest edges highlighted
    ax3.scatter(X_demo_scaled[:, 0], X_demo_scaled[:, 1], c='lightblue', s=60, 
               alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Sort edges by weight and highlight the longest ones
    sorted_indices = np.argsort(edge_weights)[::-1]
    longest_edges = sorted_indices[:3]  # Highlight 3 longest edges
    
    for i, (start, end) in enumerate(edges):
        x_coords = [X_demo_scaled[start, 0], X_demo_scaled[end, 0]]
        y_coords = [X_demo_scaled[start, 1], X_demo_scaled[end, 1]]
        
        if i in longest_edges:
            ax3.plot(x_coords, y_coords, 'r-', linewidth=3, alpha=0.8,
                    label='Longest edges' if i == longest_edges[0] else "")
        else:
            ax3.plot(x_coords, y_coords, 'k-', linewidth=1, alpha=0.5)
    
    ax3.set_title('MST with Longest Edges Highlighted', fontsize=12)
    ax3.set_xlabel('Feature 1 (scaled)')
    ax3.set_ylabel('Feature 2 (scaled)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Edge weight distribution with statistics
    ax4.hist(edge_weights, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(np.mean(edge_weights), color='green', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(edge_weights):.3f}')
    ax4.axvline(np.median(edge_weights), color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(edge_weights):.3f}')
    
    # Mark the longest edges
    for i, idx in enumerate(longest_edges):
        weight = edge_weights[idx]
        ax4.axvline(weight, color='red', linestyle='-', alpha=0.7,
                   label=f'Longest edge {i+1}: {weight:.3f}' if i < 2 else "")
    
    ax4.set_title('MST Edge Weight Distribution', fontsize=12)
    ax4.set_xlabel('Edge Weight (Distance)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save MST properties analysis
    save_path = os.path.join(output_dir, "mst_properties_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Figure saved: {save_path}")
    plt.show()
    
    # Print MST statistics
    print(f"  Total MST edges: {len(edges)}")
    print(f"  Total MST weight: {np.sum(edge_weights):.3f}")
    print(f"  Average edge weight: {np.mean(edge_weights):.3f}")
    print(f"  Median edge weight: {np.median(edge_weights):.3f}")
    print(f"  Longest edge weight: {np.max(edge_weights):.3f}")
    print(f"  Shortest edge weight: {np.min(edge_weights):.3f}")
    
    # Show clustering quality for different k values
    print("\nClustering Quality Analysis...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    k_values = range(2, 8)
    total_weights = []
    max_intra_cluster_distances = []
    
    for k in k_values:
        labels, _, _, removed_edges = mst_clustering(X_demo_scaled, n_clusters=k)
        
        # Calculate total weight of remaining MST edges
        remaining_weight = 0
        max_intra_dist = 0
        
        for i, (start, end) in enumerate(edges):
            if (start, end) not in removed_edges and (end, start) not in removed_edges:
                remaining_weight += edge_weights[i]
                if labels[start] == labels[end]:
                    max_intra_dist = max(max_intra_dist, edge_weights[i])
        
        total_weights.append(remaining_weight)
        max_intra_cluster_distances.append(max_intra_dist)
    
    # Plot total remaining MST weight vs k
    ax1.plot(k_values, total_weights, 'bo-', linewidth=2, markersize=8)
    ax1.set_title('Total MST Weight vs Number of Clusters', fontsize=12)
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Total Remaining MST Weight')
    ax1.grid(True, alpha=0.3)
    
    # Plot maximum intra-cluster distance vs k
    ax2.plot(k_values, max_intra_cluster_distances, 'ro-', linewidth=2, markersize=8)
    ax2.set_title('Max Intra-cluster Distance vs Number of Clusters', fontsize=12)
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Maximum Intra-cluster Distance')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save clustering quality analysis
    save_path = os.path.join(output_dir, "mst_clustering_quality.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Figure saved: {save_path}")
    plt.show()
    
    print("\nMST Clustering Demo Complete!")
    print(f"\nAll figures saved in the '{output_dir}' directory:")
    
    # List all saved files
    saved_files = [
        "mst_blob_data.png",
        "mst_weights_blob_data.png",
        "mst_circle_data.png", 
        "mst_weights_circle_data.png",
        "mst_moon_data.png",
        "mst_weights_moon_data.png",
        "mst_spiral_data.png",
        "mst_weights_spiral_data.png",
        "mst_comparison_blob_data.png",
        "mst_properties_analysis.png",
        "mst_clustering_quality.png"
    ]
    
    for filename in saved_files:
        print(f"  - {filename}")
    
    print("\nMST Clustering Key Concepts:")
    print("  • MST connects all points with minimum total edge weight")
    print("  • Clustering by removing longest MST edges")
    print("  • Works well for non-spherical clusters")
    print("  • Single-linkage hierarchical clustering equivalent")
    print("  • Sensitive to noise and outliers")
    print("  • Good for finding natural cluster boundaries")

if __name__ == "__main__":
    main()
