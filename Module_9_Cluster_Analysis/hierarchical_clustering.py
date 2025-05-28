import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import os

class HierarchicalClusteringDemo:
    def __init__(self, n_clusters=3, n_samples=150, random_state=42, save_figures=True):
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.random_state = random_state
        self.save_figures = save_figures
        self.colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        self.linkage_methods = ['ward', 'complete', 'average', 'single']
        
        # Create output directory for figures
        if self.save_figures:
            self.output_dir = 'hierarchical_figures'
            os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_data(self):
        """Generate random clustered data"""
        X, y_true = make_blobs(n_samples=self.n_samples, 
                              centers=self.n_clusters, 
                              cluster_std=1.2,
                              random_state=self.random_state)
        return X, y_true
    
    def manual_agglomerative_clustering(self, X, linkage_method='ward'):
        """Manual implementation of agglomerative clustering to show steps"""
        n_points = len(X)
        
        # Initialize each point as its own cluster
        clusters = {i: [i] for i in range(n_points)}
        cluster_centers = {i: X[i] for i in range(n_points)}
        
        merge_history = []
        current_cluster_id = n_points
        
        while len(clusters) > 1:
            min_distance = float('inf')
            merge_pair = None
            
            # Find closest pair of clusters
            cluster_ids = list(clusters.keys())
            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    id1, id2 = cluster_ids[i], cluster_ids[j]
                    
                    if linkage_method == 'single':
                        # Single linkage: minimum distance between any two points
                        dist = self._single_linkage_distance(X, clusters[id1], clusters[id2])
                    elif linkage_method == 'complete':
                        # Complete linkage: maximum distance between any two points
                        dist = self._complete_linkage_distance(X, clusters[id1], clusters[id2])
                    elif linkage_method == 'average':
                        # Average linkage: average distance between all pairs
                        dist = self._average_linkage_distance(X, clusters[id1], clusters[id2])
                    else:  # ward
                        # Ward linkage: minimize within-cluster variance
                        dist = self._ward_linkage_distance(X, clusters[id1], clusters[id2])
                    
                    if dist < min_distance:
                        min_distance = dist
                        merge_pair = (id1, id2)
            
            # Merge the closest clusters
            id1, id2 = merge_pair
            new_cluster = clusters[id1] + clusters[id2]
            new_center = np.mean(X[new_cluster], axis=0)
            
            # Record merge
            merge_history.append({
                'step': len(merge_history),
                'merged_clusters': (id1, id2),
                'new_cluster_id': current_cluster_id,
                'distance': min_distance,
                'clusters_snapshot': clusters.copy(),
                'n_clusters': len(clusters) - 1
            })
            
            # Update clusters
            del clusters[id1]
            del clusters[id2]
            clusters[current_cluster_id] = new_cluster
            cluster_centers[current_cluster_id] = new_center
            
            current_cluster_id += 1
        
        return merge_history
    
    def _single_linkage_distance(self, X, cluster1, cluster2):
        """Calculate single linkage distance"""
        min_dist = float('inf')
        for i in cluster1:
            for j in cluster2:
                dist = np.linalg.norm(X[i] - X[j])
                min_dist = min(min_dist, dist)
        return min_dist
    
    def _complete_linkage_distance(self, X, cluster1, cluster2):
        """Calculate complete linkage distance"""
        max_dist = 0
        for i in cluster1:
            for j in cluster2:
                dist = np.linalg.norm(X[i] - X[j])
                max_dist = max(max_dist, dist)
        return max_dist
    
    def _average_linkage_distance(self, X, cluster1, cluster2):
        """Calculate average linkage distance"""
        total_dist = 0
        count = 0
        for i in cluster1:
            for j in cluster2:
                total_dist += np.linalg.norm(X[i] - X[j])
                count += 1
        return total_dist / count
    
    def _ward_linkage_distance(self, X, cluster1, cluster2):
        """Calculate Ward linkage distance"""
        center1 = np.mean(X[cluster1], axis=0)
        center2 = np.mean(X[cluster2], axis=0)
        combined_center = np.mean(X[cluster1 + cluster2], axis=0)
        
        # Calculate increase in within-cluster sum of squares
        n1, n2 = len(cluster1), len(cluster2)
        return (n1 * n2) / (n1 + n2) * np.linalg.norm(center1 - center2) ** 2
    
    def plot_clustering_steps(self, X, merge_history, linkage_method='ward'):
        """Plot the hierarchical clustering process step by step"""
        # Show key steps in the clustering process
        steps_to_show = [0, len(merge_history)//4, len(merge_history)//2, 
                        3*len(merge_history)//4, len(merge_history)-1]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Plot initial state
        ax = axes[0]
        ax.scatter(X[:, 0], X[:, 1], c='black', s=50, alpha=0.7)
        ax.set_title(f'Initial State\n({len(X)} clusters)')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        
        # Plot clustering steps
        for idx, step in enumerate(steps_to_show[1:], 1):
            ax = axes[idx]
            clusters = merge_history[step]['clusters_snapshot']
            n_clusters = merge_history[step]['n_clusters']
            
            for cluster_id, points in clusters.items():
                color = self.colors[cluster_id % len(self.colors)]
                ax.scatter(X[points, 0], X[points, 1], c=color, s=50, alpha=0.7,
                          label=f'Cluster {cluster_id}')
            
            ax.set_title(f'Step {step + 1}\n({n_clusters} clusters)')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.grid(True, alpha=0.3)
        
        # Plot final result with sklearn
        ax = axes[5]
        sklearn_clustering = AgglomerativeClustering(n_clusters=self.n_clusters, 
                                                   linkage=linkage_method)
        sklearn_labels = sklearn_clustering.fit_predict(X)
        
        for i in range(self.n_clusters):
            mask = sklearn_labels == i
            ax.scatter(X[mask, 0], X[mask, 1], c=self.colors[i], s=50, alpha=0.7,
                      label=f'Final Cluster {i+1}')
        
        ax.set_title(f'Final Result\n({self.n_clusters} clusters)')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.suptitle(f'Hierarchical Clustering Process ({linkage_method.title()} Linkage)', 
                    fontsize=16, y=0.98)
        
        if self.save_figures:
            filename = f'{self.output_dir}/hierarchical_process_{linkage_method}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        plt.show()
    
    def plot_dendrograms(self, X):
        """Plot dendrograms for different linkage methods"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, method in enumerate(self.linkage_methods):
            ax = axes[idx]
            
            # Compute linkage matrix
            linkage_matrix = linkage(X, method=method)
            
            # Plot dendrogram
            dendrogram(linkage_matrix, ax=ax, truncate_mode='level', p=10,
                      color_threshold=0.7*max(linkage_matrix[:,2]))
            
            ax.set_title(f'{method.title()} Linkage Dendrogram')
            ax.set_xlabel('Sample Index or (Cluster Size)')
            ax.set_ylabel('Distance')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Dendrograms for Different Linkage Methods', fontsize=16, y=0.98)
        
        if self.save_figures:
            filename = f'{self.output_dir}/dendrograms_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        plt.show()
    
    def plot_linkage_comparison(self, X):
        """Compare different linkage methods"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, method in enumerate(self.linkage_methods):
            ax = axes[idx]
            
            # Perform clustering
            clustering = AgglomerativeClustering(n_clusters=self.n_clusters, 
                                               linkage=method)
            labels = clustering.fit_predict(X)
            
            # Plot results
            for i in range(self.n_clusters):
                mask = labels == i
                ax.scatter(X[mask, 0], X[mask, 1], c=self.colors[i], s=50, alpha=0.7,
                          label=f'Cluster {i+1}')
            
            ax.set_title(f'{method.title()} Linkage')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.suptitle('Comparison of Linkage Methods', fontsize=16, y=0.98)
        
        if self.save_figures:
            filename = f'{self.output_dir}/linkage_methods_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        plt.show()
    
    def plot_distance_matrix(self, X):
        """Plot distance matrix heatmap"""
        # Calculate pairwise distances
        distances = pdist(X)
        distance_matrix = squareform(distances)
        
        # Create hierarchical clustering and reorder matrix
        linkage_matrix = linkage(X, method='ward')
        dendro = dendrogram(linkage_matrix, no_plot=True)
        reorder_idx = dendro['leaves']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Original distance matrix
        im1 = ax1.imshow(distance_matrix, cmap='viridis', aspect='auto')
        ax1.set_title('Original Distance Matrix')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Sample Index')
        plt.colorbar(im1, ax=ax1, label='Distance')
        
        # Reordered distance matrix
        reordered_matrix = distance_matrix[np.ix_(reorder_idx, reorder_idx)]
        im2 = ax2.imshow(reordered_matrix, cmap='viridis', aspect='auto')
        ax2.set_title('Reordered Distance Matrix\n(Hierarchical Clustering Order)')
        ax2.set_xlabel('Sample Index (Reordered)')
        ax2.set_ylabel('Sample Index (Reordered)')
        plt.colorbar(im2, ax=ax2, label='Distance')
        
        plt.tight_layout()
        
        if self.save_figures:
            filename = f'{self.output_dir}/distance_matrix_analysis.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        plt.show()
    
    def plot_cluster_validation(self, X, y_true):
        """Plot cluster validation metrics for different numbers of clusters"""
        from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
        
        max_clusters = min(10, len(X) - 1)
        n_clusters_range = range(2, max_clusters + 1)
        
        metrics = {
            'silhouette': [],
            'calinski_harabasz': [],
            'ari': []
        }
        
        for n_clust in n_clusters_range:
            clustering = AgglomerativeClustering(n_clusters=n_clust, linkage='ward')
            labels = clustering.fit_predict(X)
            
            metrics['silhouette'].append(silhouette_score(X, labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
            metrics['ari'].append(adjusted_rand_score(y_true, labels))
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Silhouette Score
        axes[0].plot(n_clusters_range, metrics['silhouette'], 'bo-', linewidth=2, markersize=8)
        axes[0].axvline(x=self.n_clusters, color='red', linestyle='--', alpha=0.7, label='True k')
        axes[0].set_title('Silhouette Score')
        axes[0].set_xlabel('Number of Clusters')
        axes[0].set_ylabel('Silhouette Score')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Calinski-Harabasz Score
        axes[1].plot(n_clusters_range, metrics['calinski_harabasz'], 'go-', linewidth=2, markersize=8)
        axes[1].axvline(x=self.n_clusters, color='red', linestyle='--', alpha=0.7, label='True k')
        axes[1].set_title('Calinski-Harabasz Score')
        axes[1].set_xlabel('Number of Clusters')
        axes[1].set_ylabel('Calinski-Harabasz Score')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Adjusted Rand Index
        axes[2].plot(n_clusters_range, metrics['ari'], 'ro-', linewidth=2, markersize=8)
        axes[2].axvline(x=self.n_clusters, color='red', linestyle='--', alpha=0.7, label='True k')
        axes[2].set_title('Adjusted Rand Index')
        axes[2].set_xlabel('Number of Clusters')
        axes[2].set_ylabel('ARI Score')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        plt.suptitle('Cluster Validation Metrics', fontsize=16, y=1.02)
        
        if self.save_figures:
            filename = f'{self.output_dir}/cluster_validation_metrics.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        plt.show()
    
    def create_detailed_dendrogram(self, X, method='ward'):
        """Create a detailed dendrogram with cluster highlighting"""
        linkage_matrix = linkage(X, method=method)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Full dendrogram
        dendro1 = dendrogram(linkage_matrix, ax=ax1, 
                           color_threshold=0.7*max(linkage_matrix[:,2]),
                           above_threshold_color='gray')
        ax1.set_title(f'Complete Dendrogram ({method.title()} Linkage)')
        ax1.set_xlabel('Sample Index or (Cluster Size)')
        ax1.set_ylabel('Distance')
        ax1.grid(True, alpha=0.3)
        
        # Truncated dendrogram with cluster cut
        threshold = linkage_matrix[-self.n_clusters+1, 2]
        dendro2 = dendrogram(linkage_matrix, ax=ax2, 
                           truncate_mode='lastp', p=20,
                           color_threshold=threshold,
                           above_threshold_color='gray')
        ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Cut for {self.n_clusters} clusters')
        ax2.set_title(f'Truncated Dendrogram with Cluster Cut')
        ax2.set_xlabel('Sample Index or (Cluster Size)')
        ax2.set_ylabel('Distance')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if self.save_figures:
            filename = f'{self.output_dir}/detailed_dendrogram_{method}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        plt.show()
    
    def save_individual_steps(self, X, merge_history, linkage_method='ward'):
        """Save individual clustering steps as separate images"""
        if not self.save_figures:
            return
            
        steps_dir = f'{self.output_dir}/clustering_steps_{linkage_method}'
        os.makedirs(steps_dir, exist_ok=True)
        
        # Save initial state
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(X[:, 0], X[:, 1], c='black', s=50, alpha=0.7)
        ax.set_title(f'Initial State - {len(X)} Individual Points')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        filename = f'{steps_dir}/step_000_initial.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save key steps
        step_interval = max(1, len(merge_history) // 10)
        for i in range(0, len(merge_history), step_interval):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            clusters = merge_history[i]['clusters_snapshot']
            n_clusters = merge_history[i]['n_clusters']
            
            for cluster_id, points in clusters.items():
                color = self.colors[cluster_id % len(self.colors)]
                ax.scatter(X[points, 0], X[points, 1], c=color, s=50, alpha=0.7)
            
            ax.set_title(f'Step {i+1} - {n_clusters} Clusters')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.grid(True, alpha=0.3)
            
            filename = f'{steps_dir}/step_{i+1:03d}_clusters_{n_clusters}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Individual steps saved in: {steps_dir}")
    
    def compare_with_kmeans(self, X):
        """Compare hierarchical clustering with K-means"""
        from sklearn.cluster import KMeans
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        
        for i in range(self.n_clusters):
            mask = kmeans_labels == i
            axes[0].scatter(X[mask, 0], X[mask, 1], c=self.colors[i], s=50, alpha=0.7,
                          label=f'Cluster {i+1}')
        axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                       c='black', marker='x', s=200, linewidths=3, label='Centroids')
        axes[0].set_title('K-Means Clustering')
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Hierarchical clustering (Ward)
        hierarchical = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward')
        hierarchical_labels = hierarchical.fit_predict(X)
        
        for i in range(self.n_clusters):
            mask = hierarchical_labels == i
            axes[1].scatter(X[mask, 0], X[mask, 1], c=self.colors[i], s=50, alpha=0.7,
                          label=f'Cluster {i+1}')
        axes[1].set_title('Hierarchical Clustering (Ward)')
        axes[1].set_xlabel('Feature 1')
        axes[1].set_ylabel('Feature 2')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Difference visualization
        different_points = kmeans_labels != hierarchical_labels
        axes[2].scatter(X[~different_points, 0], X[~different_points, 1], 
                       c='lightgray', s=30, alpha=0.5, label='Same cluster')
        axes[2].scatter(X[different_points, 0], X[different_points, 1], 
                       c='red', s=50, alpha=0.8, label='Different cluster')
        axes[2].set_title('Clustering Differences\n(Red = Different assignments)')
        axes[2].set_xlabel('Feature 1')
        axes[2].set_ylabel('Feature 2')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        
        if self.save_figures:
            filename = f'{self.output_dir}/hierarchical_vs_kmeans.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        plt.show()
        
        # Print comparison metrics
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ari = adjusted_rand_score(kmeans_labels, hierarchical_labels)
        nmi = normalized_mutual_info_score(kmeans_labels, hierarchical_labels)
        
        print(f"\nComparison between K-Means and Hierarchical Clustering:")
        print(f"Adjusted Rand Index: {ari:.3f}")
        print(f"Normalized Mutual Information: {nmi:.3f}")
    
    def run_demo(self):
        """Run the complete hierarchical clustering demo"""
        print("Hierarchical Clustering Demo")
        print("=" * 60)
        
        # Generate data
        print(f"Generating {self.n_samples} data points with {self.n_clusters} natural clusters...")
        X, y_true = self.generate_data()
        
        if self.save_figures:
            print(f"Figures will be saved to: {self.output_dir}/")
        
        # Run manual hierarchical clustering
        print("\nRunning manual agglomerative clustering (Ward linkage)...")
        merge_history = self.manual_agglomerative_clustering(X, 'ward')
        print(f"Clustering completed in {len(merge_history)} merge steps")
        
        # Create visualizations
        print("\nCreating visualizations...")
        
        # 1. Show clustering process
        self.plot_clustering_steps(X, merge_history, 'ward')
        
        # 2. Compare different linkage methods
        print("Comparing different linkage methods...")
        self.plot_linkage_comparison(X)
        
        # 3. Show dendrograms
        print("Creating dendrograms...")
        self.plot_dendrograms(X)
        
        # 4. Detailed dendrogram analysis
        print("Creating detailed dendrogram analysis...")
        self.create_detailed_dendrogram(X, 'ward')
        
        # 5. Distance matrix analysis
        print("Analyzing distance matrix...")
        self.plot_distance_matrix(X)
        
        # 6. Cluster validation
        print("Performing cluster validation...")
        self.plot_cluster_validation(X, y_true)
        
        # 7. Compare with K-means
        print("Comparing with K-means clustering...")
        self.compare_with_kmeans(X)
        
        # Save individual steps
        if self.save_figures:
            print("Saving individual clustering steps...")
            self.save_individual_steps(X, merge_history, 'ward')
        
        # Final metrics
        print("\nFinal Analysis:")
        hierarchical = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward')
        final_labels = hierarchical.fit_predict(X)
        
        from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
        
        silhouette_avg = silhouette_score(X, final_labels)
        ari_score = adjusted_rand_score(y_true, final_labels)
        ch_score = calinski_harabasz_score(X, final_labels)
        
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"Adjusted Rand Index: {ari_score:.3f}")
        print(f"Calinski-Harabasz Score: {ch_score:.3f}")
        
        if self.save_figures:
            print(f"\nAll figures saved to: {os.path.abspath(self.output_dir)}/")

def main():
    """Main function to run the demo"""
    # Create and run the demo
    demo = HierarchicalClusteringDemo(n_clusters=3, n_samples=150, random_state=42, save_figures=True)
    demo.run_demo()
    
    # Optional: Run with different parameters
    print("\n" + "="*60)
    print("Running demo with different parameters (4 clusters)...")
    demo2 = HierarchicalClusteringDemo(n_clusters=4, n_samples=200, random_state=123, save_figures=True)
    demo2.run_demo()

if __name__ == "__main__":
    main()
