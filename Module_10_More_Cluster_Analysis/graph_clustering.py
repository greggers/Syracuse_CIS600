import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
import networkx as nx
from scipy.sparse import csgraph
from scipy.linalg import eigh
import os

# Create output directory for figures
output_dir = "graph_clustering_figures"
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
    elif dataset_type == 'swiss_roll':
        # Create a 2D version of swiss roll
        t = np.linspace(0, 4*np.pi, n_samples)
        x = t * np.cos(t) + np.random.normal(0, 0.3, n_samples)
        y = t * np.sin(t) + np.random.normal(0, 0.3, n_samples)
        X = np.column_stack([x, y])
    else:
        raise ValueError("dataset_type must be 'blobs', 'circles', 'moons', or 'swiss_roll'")
    
    return X

def build_similarity_graph(X, method='knn', k=10, sigma=1.0, eps=0.5):
    """Build similarity graph from data points"""
    if method == 'knn':
        # k-nearest neighbors graph
        adjacency = kneighbors_graph(X, n_neighbors=k, mode='distance', 
                                   include_self=False)
        # Convert to similarity (inverse distance) and make symmetric
        adjacency.data = 1.0 / (adjacency.data + 1e-10)
        adjacency = (adjacency + adjacency.T) / 2
        adjacency = adjacency.toarray()
        
    elif method == 'epsilon':
        # Epsilon-neighborhood graph with distance weights
        adjacency = radius_neighbors_graph(X, radius=eps, mode='distance',
                                         include_self=False)
        # Convert to similarity
        adjacency.data = 1.0 / (adjacency.data + 1e-10)
        adjacency = adjacency.toarray()
        
    elif method == 'rbf':
        # RBF (Gaussian) kernel similarity
        similarity = rbf_kernel(X, gamma=1/(2*sigma**2))
        # Remove self-connections and apply threshold
        np.fill_diagonal(similarity, 0)
        adjacency = similarity * (similarity > 0.1)
        
    elif method == 'full_rbf':
        # Full RBF kernel (weighted graph)
        similarity = rbf_kernel(X, gamma=1/(2*sigma**2))
        np.fill_diagonal(similarity, 0)
        adjacency = similarity
        
    return adjacency

def spectral_clustering_custom(X, n_clusters=2, graph_method='knn', k=10, sigma=1.0):
    """Custom implementation of spectral clustering"""
    # Build similarity graph
    W = build_similarity_graph(X, method=graph_method, k=k, sigma=sigma)
    
    # Ensure W is a numpy array
    if hasattr(W, 'toarray'):
        W = W.toarray()
    
    # Compute degree matrix
    D = np.diag(np.sum(W, axis=1))
    
    # Compute normalized Laplacian: L = D^(-1/2) * (D - W) * D^(-1/2)
    D_sqrt_inv = np.diag(1.0 / np.sqrt(np.sum(W, axis=1) + 1e-10))
    L_norm = D_sqrt_inv @ (D - W) @ D_sqrt_inv
    
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = eigh(L_norm)
    
    # Use the first n_clusters eigenvectors (smallest eigenvalues)
    embedding = eigenvecs[:, :n_clusters]
    
    # Normalize rows
    embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-10)
    
    # Apply k-means to the embedding
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embedding)
    
    return labels, W, embedding, eigenvals

def graph_cut_clustering(X, n_clusters=2, method='normalized_cut'):
    """Perform graph cut clustering"""
    # Build similarity graph
    W = build_similarity_graph(X, method='rbf', sigma=1.0)
    if hasattr(W, 'toarray'):
        W = W.toarray()
    
    # Compute degree matrix
    D = np.diag(np.sum(W, axis=1))
    
    if method == 'normalized_cut':
        # Normalized cut: solve generalized eigenvalue problem
        L = D - W  # Laplacian matrix
        eigenvals, eigenvecs = eigh(L, D + 1e-10 * np.eye(len(D)))
        
    elif method == 'ratio_cut':
        # Ratio cut: use normalized Laplacian
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
        L_norm = D_inv_sqrt @ (D - W) @ D_inv_sqrt
        eigenvals, eigenvecs = eigh(L_norm)
    
    # Use second smallest eigenvector for 2-way cut
    if n_clusters == 2:
        fiedler_vector = eigenvecs[:, 1]
        labels = (fiedler_vector > np.median(fiedler_vector)).astype(int)
    else:
        # For multiple clusters, use multiple eigenvectors
        embedding = eigenvecs[:, 1:n_clusters+1]
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embedding)
    
    return labels, W, eigenvecs[:, 1:n_clusters+1]

def community_detection_clustering(X, method='louvain'):
    """Perform community detection on similarity graph"""
    # Build similarity graph with weights
    W = build_similarity_graph(X, method='knn', k=8)
    if hasattr(W, 'toarray'):
        W = W.toarray()
    
    # Create NetworkX graph
    G = nx.from_numpy_array(W)
    
    if method == 'louvain':
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G)
            labels = np.array([partition[i] for i in range(len(X))])
        except ImportError:
            # Fallback to NetworkX community detection
            communities = nx.community.greedy_modularity_communities(G)
            labels = np.zeros(len(X))
            for i, community in enumerate(communities):
                for node in community:
                    labels[node] = i
    
    elif method == 'modularity':
        communities = nx.community.greedy_modularity_communities(G)
        labels = np.zeros(len(X))
        for i, community in enumerate(communities):
            for node in community:
                labels[node] = i
    
    return labels.astype(int), W, G

def plot_graph_clustering_results(X, labels, W, title, method_name, save_name=None):
    """Plot graph clustering results with the similarity graph"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define colors for clusters
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    # Plot 1: Original data
    ax1.scatter(X[:, 0], X[:, 1], c='lightblue', s=60, alpha=0.8, 
               edgecolors='black', linewidth=0.5)
    ax1.set_title('Original Data', fontsize=12)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Similarity graph
    ax2.scatter(X[:, 0], X[:, 1], c='lightblue', s=60, alpha=0.8, 
               edgecolors='black', linewidth=0.5)
    
    # Draw edges of similarity graph (sample a subset for clarity)
    n_points = len(X)
    
    # Fix: Handle the case where W might be boolean or have no positive values
    W_positive = W[W > 0]
    if len(W_positive) > 0:
        # Convert boolean to float if necessary
        if W_positive.dtype == bool:
            W_positive = W_positive.astype(float)
        edge_threshold = np.percentile(W_positive, 80)
    else:
        edge_threshold = 0
    
    edges_drawn = 0
    max_edges = 100  # Limit number of edges for clarity
    
    for i in range(n_points):
        for j in range(i+1, n_points):
            if W[i, j] > edge_threshold and edges_drawn < max_edges:
                ax2.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 
                        'k-', alpha=0.3, linewidth=0.5)
                edges_drawn += 1
    
    ax2.set_title(f'Similarity Graph\n({method_name})', fontsize=12)
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Clustering results
    for i, color in enumerate(colors):
        mask = labels == unique_labels[i]
        if np.any(mask):
            ax3.scatter(X[mask, 0], X[mask, 1], c=[color], s=60, 
                       alpha=0.8, edgecolors='black', linewidth=0.5,
                       label=f'Cluster {unique_labels[i]+1}')
    
    n_clusters = len(unique_labels)
    ax3.set_title(f'Clustering Result\n{n_clusters} clusters found', fontsize=12)
    ax3.set_xlabel('Feature 1')
    ax3.set_ylabel('Feature 2')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'{title} - {method_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure if save_name is provided
    if save_name:
        save_path = os.path.join(output_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved: {save_path}")
    
    return fig

def plot_spectral_embedding(X, labels, embedding, eigenvals, title, save_name=None):
    """Plot spectral embedding and eigenvalues"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define colors for clusters
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    # Plot 1: Original data with clustering
    for i, color in enumerate(colors):
        mask = labels == unique_labels[i]
        if np.any(mask):
            ax1.scatter(X[mask, 0], X[mask, 1], c=[color], s=60, 
                       alpha=0.8, edgecolors='black', linewidth=0.5,
                       label=f'Cluster {unique_labels[i]+1}')
    
    ax1.set_title('Original Data with Clusters', fontsize=12)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Spectral embedding (first 2 eigenvectors)
    if embedding.shape[1] >= 2:
        for i, color in enumerate(colors):
            mask = labels == unique_labels[i]
            if np.any(mask):
                ax2.scatter(embedding[mask, 0], embedding[mask, 1], c=[color], s=60, 
                           alpha=0.8, edgecolors='black', linewidth=0.5)
        
        ax2.set_title('Spectral Embedding\n(First 2 Eigenvectors)', fontsize=12)
        ax2.set_xlabel('1st Eigenvector')
        ax2.set_ylabel('2nd Eigenvector')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Not enough eigenvectors\nfor 2D visualization', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Spectral Embedding', fontsize=12)
    
        # Plot 3: Eigenvalues
    ax3.plot(range(1, min(21, len(eigenvals)+1)), eigenvals[:20], 'bo-', linewidth=2, markersize=6)
    ax3.set_title('Eigenvalues of Graph Laplacian', fontsize=12)
    ax3.set_xlabel('Eigenvalue Index')
    ax3.set_ylabel('Eigenvalue')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Eigenvector components
    if embedding.shape[1] >= 1:
        ax4.plot(embedding[:, 0], 'b-', linewidth=1, alpha=0.7, label='1st Eigenvector')
        if embedding.shape[1] >= 2:
            ax4.plot(embedding[:, 1], 'r-', linewidth=1, alpha=0.7, label='2nd Eigenvector')
        if embedding.shape[1] >= 3:
            ax4.plot(embedding[:, 2], 'g-', linewidth=1, alpha=0.7, label='3rd Eigenvector')
        
        ax4.set_title('Eigenvector Components', fontsize=12)
        ax4.set_xlabel('Data Point Index')
        ax4.set_ylabel('Eigenvector Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{title} - Spectral Analysis', fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save figure if save_name is provided
    if save_name:
        save_path = os.path.join(output_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved: {save_path}")
    
    return fig

def compare_graph_methods(X, dataset_name, n_clusters=3):
    """Compare different graph clustering methods"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    methods = [
        ('Spectral (KNN)', 'spectral_knn'),
        ('Spectral (RBF)', 'spectral_rbf'), 
        ('Normalized Cut', 'normalized_cut'),
        ('Ratio Cut', 'ratio_cut'),
        ('Community Detection', 'community'),
        ('Sklearn Spectral', 'sklearn_spectral')
    ]
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    for idx, (method_name, method_key) in enumerate(methods):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        try:
            if method_key == 'spectral_knn':
                labels, W, embedding, eigenvals = spectral_clustering_custom(
                    X, n_clusters=n_clusters, graph_method='knn', k=8)
            elif method_key == 'spectral_rbf':
                labels, W, embedding, eigenvals = spectral_clustering_custom(
                    X, n_clusters=n_clusters, graph_method='rbf', sigma=1.0)
            elif method_key == 'normalized_cut':
                labels, W, embedding = graph_cut_clustering(
                    X, n_clusters=n_clusters, method='normalized_cut')
            elif method_key == 'ratio_cut':
                labels, W, embedding = graph_cut_clustering(
                    X, n_clusters=n_clusters, method='ratio_cut')
            elif method_key == 'community':
                labels, W, G = community_detection_clustering(X, method='modularity')
            elif method_key == 'sklearn_spectral':
                spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
                                            n_neighbors=8, random_state=42)
                labels = spectral.fit_predict(X)
            
            # Plot results
            unique_labels = np.unique(labels)
            for i, label in enumerate(unique_labels):
                mask = labels == label
                if np.any(mask):
                    color_idx = i % len(colors)
                    ax.scatter(X[mask, 0], X[mask, 1], c=[colors[color_idx]], s=50, 
                             alpha=0.8, edgecolors='black', linewidth=0.5)
            
            ax.set_title(f'{method_name}\n{len(unique_labels)} clusters', fontsize=11)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{method_name}\nError', fontsize=11)
    
    plt.suptitle(f'Graph Clustering Methods Comparison - {dataset_name}', fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save comparison figure
    save_name = f"graph_methods_comparison_{dataset_name.lower().replace(' ', '_')}"
    save_path = os.path.join(output_dir, f"{save_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Figure saved: {save_path}")
    
    return fig

def analyze_graph_parameters(X, dataset_name):
    """Analyze the effect of different graph construction parameters"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Test different k values for KNN graph
    k_values = [3, 5, 8, 12, 15]
    colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))
    
    for i, k in enumerate(k_values):
        try:
            labels, W, embedding, eigenvals = spectral_clustering_custom(
                X, n_clusters=3, graph_method='knn', k=k)
            
            # Plot first few eigenvalues
            ax1.plot(range(1, min(11, len(eigenvals)+1)), eigenvals[:10], 
                    'o-', color=colors[i], label=f'k={k}', alpha=0.7)
        except Exception as e:
            print(f"  Warning: Failed for k={k}: {str(e)[:50]}")
            continue
    
    ax1.set_title('Eigenvalues vs KNN Parameter k', fontsize=12)
    ax1.set_xlabel('Eigenvalue Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test different sigma values for RBF graph
    sigma_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    colors = plt.cm.plasma(np.linspace(0, 1, len(sigma_values)))
    
    for i, sigma in enumerate(sigma_values):
        try:
            labels, W, embedding, eigenvals = spectral_clustering_custom(
                X, n_clusters=3, graph_method='rbf', sigma=sigma)
            
            ax2.plot(range(1, min(11, len(eigenvals)+1)), eigenvals[:10], 
                    'o-', color=colors[i], label=f'σ={sigma}', alpha=0.7)
        except Exception as e:
            print(f"  Warning: Failed for sigma={sigma}: {str(e)[:50]}")
            continue
    
    ax2.set_title('Eigenvalues vs RBF Parameter σ', fontsize=12)
    ax2.set_xlabel('Eigenvalue Index')
    ax2.set_ylabel('Eigenvalue')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Show connectivity patterns for different k values
    k_test = 8
    try:
        W_knn = build_similarity_graph(X, method='knn', k=k_test)
        if hasattr(W_knn, 'toarray'):
            W_knn = W_knn.toarray()
        
        # Plot connectivity matrix
        im1 = ax3.imshow(W_knn, cmap='Blues', aspect='auto')
        ax3.set_title(f'KNN Adjacency Matrix (k={k_test})', fontsize=12)
        ax3.set_xlabel('Data Point Index')
        ax3.set_ylabel('Data Point Index')
        plt.colorbar(im1, ax=ax3)
    except Exception as e:
        ax3.text(0.5, 0.5, f'Error creating\nKNN matrix:\n{str(e)[:30]}...', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title(f'KNN Adjacency Matrix (k={k_test})', fontsize=12)
    
    # Show RBF similarity matrix
    try:
        W_rbf = build_similarity_graph(X, method='full_rbf', sigma=1.0)
        if hasattr(W_rbf, 'toarray'):
            W_rbf = W_rbf.toarray()
        
        im2 = ax4.imshow(W_rbf, cmap='Reds', aspect='auto')
        ax4.set_title('RBF Similarity Matrix (σ=1.0)', fontsize=12)
        ax4.set_xlabel('Data Point Index')
        ax4.set_ylabel('Data Point Index')
        plt.colorbar(im2, ax=ax4)
    except Exception as e:
        ax4.text(0.5, 0.5, f'Error creating\nRBF matrix:\n{str(e)[:30]}...', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('RBF Similarity Matrix (σ=1.0)', fontsize=12)
    
    plt.suptitle(f'Graph Construction Parameter Analysis - {dataset_name}', fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save parameter analysis
    save_name = f"graph_parameters_{dataset_name.lower().replace(' ', '_')}"
    save_path = os.path.join(output_dir, f"{save_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Figure saved: {save_path}")
    
    return fig

def main():
    """Main function to demonstrate graph clustering"""
    print("Graph Clustering Demo")
    print("=" * 50)
    
    # Generate different datasets
    datasets = {
        'Blob Data': generate_sample_data('blobs', n_samples=150),
        'Circle Data': generate_sample_data('circles', n_samples=150, noise=0.08),
        'Moon Data': generate_sample_data('moons', n_samples=150, noise=0.08),
        'Swiss Roll': generate_sample_data('swiss_roll', n_samples=120)
    }
    
    # Optimal parameters for each dataset
    optimal_params = {
        'Blob Data': {'n_clusters': 4, 'method': 'knn', 'k': 8},
        'Circle Data': {'n_clusters': 2, 'method': 'rbf', 'sigma': 1.0},
        'Moon Data': {'n_clusters': 2, 'method': 'knn', 'k': 10},
        'Swiss Roll': {'n_clusters': 3, 'method': 'rbf', 'sigma': 1.5}
    }
    
    # Process each dataset with spectral clustering
    for dataset_name, X in datasets.items():
        print(f"\nProcessing {dataset_name} with Spectral Clustering...")
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        params = optimal_params[dataset_name]
        
        try:
            # Perform spectral clustering
            if params['method'] == 'knn':
                labels, W, embedding, eigenvals = spectral_clustering_custom(
                    X_scaled, n_clusters=params['n_clusters'], 
                    graph_method='knn', k=params['k'])
            else:
                labels, W, embedding, eigenvals = spectral_clustering_custom(
                    X_scaled, n_clusters=params['n_clusters'], 
                    graph_method='rbf', sigma=params['sigma'])
            
            # Print statistics
            unique_labels = np.unique(labels)
            print(f"  Number of clusters found: {len(unique_labels)}")
            for i in unique_labels:
                count = np.sum(labels == i)
                print(f"  Cluster {i+1}: {count} points")
            
            # Plot main results
            save_name = f"spectral_{dataset_name.lower().replace(' ', '_')}"
            fig1 = plot_graph_clustering_results(X_scaled, labels, W, 
                                               f'Spectral Clustering - {dataset_name}',
                                               f"{params['method'].upper()} Graph", save_name)
            plt.show()
            
            # Plot spectral embedding analysis
            save_name_spectral = f"spectral_analysis_{dataset_name.lower().replace(' ', '_')}"
            fig2 = plot_spectral_embedding(X_scaled, labels, embedding, eigenvals,
                                         f'Spectral Clustering - {dataset_name}', save_name_spectral)
            plt.show()
            
        except Exception as e:
            print(f"  Error processing {dataset_name}: {str(e)}")
            continue
    
    # Compare different graph clustering methods
    print("\nComparing Graph Clustering Methods...")
    for dataset_name, X in list(datasets.items())[:2]:  # Test on first 2 datasets
        print(f"  Comparing methods on {dataset_name}...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        try:
            n_clusters = optimal_params[dataset_name]['n_clusters']
            fig3 = compare_graph_methods(X_scaled, dataset_name, n_clusters)
            plt.show()
        except Exception as e:
            print(f"  Error comparing methods on {dataset_name}: {str(e)}")
            continue
    
    # Parameter analysis
    print("\nAnalyzing Graph Construction Parameters...")
    X_test = datasets['Blob Data']
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    try:
        fig4 = analyze_graph_parameters(X_test_scaled, 'Blob Data')
        plt.show()
    except Exception as e:
        print(f"  Error in parameter analysis: {str(e)}")
    
    # Demonstrate graph cut clustering
    print("\nDemonstrating Graph Cut Clustering...")
    X_moons = datasets['Moon Data']
    scaler = StandardScaler()
    X_moons_scaled = scaler.fit_transform(X_moons)
    
    try:
        # Normalized cut
        labels_ncut, W_ncut, embedding_ncut = graph_cut_clustering(
            X_moons_scaled, n_clusters=2, method='normalized_cut')
        
        fig5 = plot_graph_clustering_results(X_moons_scaled, labels_ncut, W_ncut,
                                           'Graph Cut Clustering - Moon Data',
                                           'Normalized Cut', 'normalized_cut_moons')
        plt.show()
        
        print(f"  Normalized Cut found {len(np.unique(labels_ncut))} clusters")
        
    except Exception as e:
        print(f"  Error in graph cut clustering: {str(e)}")
    
    print("\nGraph Clustering Demo Complete!")
    print(f"\nAll figures saved in the '{output_dir}' directory:")
    
    # List expected saved files
    saved_files = [
        "spectral_blob_data.png",
        "spectral_analysis_blob_data.png",
        "spectral_circle_data.png",
        "spectral_analysis_circle_data.png", 
        "spectral_moon_data.png",
        "spectral_analysis_moon_data.png",
        "spectral_swiss_roll.png",
        "spectral_analysis_swiss_roll.png",
        "graph_methods_comparison_blob_data.png",
        "graph_methods_comparison_circle_data.png",
        "graph_parameters_blob_data.png",
        "normalized_cut_moons.png"
    ]
    
    # Check which files actually exist and list them
    existing_files = []
    for filename in saved_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            existing_files.append(filename)
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (not created)")
    
    print(f"\nSuccessfully created {len(existing_files)} out of {len(saved_files)} expected figures.")
    
    print("\nGraph Clustering Key Concepts:")
    print("  • Spectral Clustering: Uses eigenvectors of graph Laplacian")
    print("  • Graph Construction: KNN, ε-neighborhood, RBF similarity")
    print("  • Normalized Cut: Minimizes cut cost normalized by cluster size")
    print("  • Ratio Cut: Minimizes cut cost normalized by cluster cardinality") 
    print("  • Community Detection: Finds densely connected subgroups")
    print("  • Works well for non-convex clusters and manifold data")
    print("  • Sensitive to graph construction parameters")
    print("  • Eigenvalue gaps indicate natural number of clusters")
    
    print("\nTroubleshooting Tips:")
    print("  • If errors occur, check that all required packages are installed:")
    print("    - numpy, matplotlib, sklearn, networkx, scipy")
    print("  • For community detection, install python-louvain:")
    print("    - pip install python-louvain")
    print("  • Adjust graph parameters (k, sigma) for different datasets")
    print("  • Use smaller datasets if memory issues occur")

if __name__ == "__main__":
    main()
