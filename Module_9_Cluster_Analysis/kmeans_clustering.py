import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import os

class KMeansDemo:
    def __init__(self, n_clusters=3, n_samples=300, random_state=42, save_figures=True):
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.random_state = random_state
        self.save_figures = save_figures
        self.colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        # Create output directory for figures
        if self.save_figures:
            self.output_dir = 'kmeans_figures'
            os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_data(self):
        """Generate random clustered data"""
        X, y_true = make_blobs(n_samples=self.n_samples, 
                              centers=self.n_clusters, 
                              cluster_std=1.5,
                              random_state=self.random_state)
        return X, y_true
    
    def manual_kmeans(self, X, max_iters=10):
        """Manual implementation of K-means to show iterations"""
        # Initialize centroids randomly
        np.random.seed(self.random_state)
        centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        iterations_data = []
        
        for iteration in range(max_iters):
            # Assign points to closest centroid
            distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Store iteration data
            iterations_data.append({
                'centroids': centroids.copy(),
                'labels': labels.copy(),
                'iteration': iteration
            })
            
            # Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            
            # Check for convergence
            if np.allclose(centroids, new_centroids):
                break
                
            centroids = new_centroids
        
        return iterations_data, labels, centroids
    
    def plot_iteration(self, X, iteration_data, subplot_pos, fig):
        """Plot a single iteration"""
        ax = fig.add_subplot(2, 3, subplot_pos)
        
        centroids = iteration_data['centroids']
        labels = iteration_data['labels']
        iteration = iteration_data['iteration']
        
        # Plot data points
        for i in range(self.n_clusters):
            mask = labels == i
            ax.scatter(X[mask, 0], X[mask, 1], 
                      c=self.colors[i], alpha=0.6, s=30, 
                      label=f'Cluster {i+1}')
        
        # Plot centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                  c='black', marker='x', s=200, linewidths=3,
                  label='Centroids')
        
        ax.set_title(f'Iteration {iteration + 1}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def save_individual_iterations(self, X, iterations_data):
        """Save individual iteration plots"""
        if not self.save_figures:
            return
            
        for i, iteration_data in enumerate(iterations_data):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            centroids = iteration_data['centroids']
            labels = iteration_data['labels']
            iteration = iteration_data['iteration']
            
            # Plot data points
            for j in range(self.n_clusters):
                mask = labels == j
                ax.scatter(X[mask, 0], X[mask, 1], 
                          c=self.colors[j], alpha=0.6, s=30, 
                          label=f'Cluster {j+1}')
            
            # Plot centroids
            ax.scatter(centroids[:, 0], centroids[:, 1], 
                      c='black', marker='x', s=200, linewidths=3,
                      label='Centroids')
            
            ax.set_title(f'K-Means Clustering - Iteration {iteration + 1}')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            filename = f'{self.output_dir}/iteration_{iteration + 1:02d}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {filename}")
    
    def visualize_clustering_process(self, X, iterations_data):
        """Create visualization showing the iterative process"""
        fig = plt.figure(figsize=(18, 12))
        
        # Plot first 5 iterations
        for i, iteration_data in enumerate(iterations_data[:5]):
            self.plot_iteration(X, iteration_data, i + 1, fig)
        
        # Plot final result using sklearn
        kmeans_sklearn = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        sklearn_labels = kmeans_sklearn.fit_predict(X)
        
        ax_final = fig.add_subplot(2, 3, 6)
        for i in range(self.n_clusters):
            mask = sklearn_labels == i
            ax_final.scatter(X[mask, 0], X[mask, 1], 
                           c=self.colors[i], alpha=0.6, s=30,
                           label=f'Cluster {i+1}')
        
        ax_final.scatter(kmeans_sklearn.cluster_centers_[:, 0], 
                        kmeans_sklearn.cluster_centers_[:, 1],
                        c='black', marker='x', s=200, linewidths=3,
                        label='Final Centroids')
        
        ax_final.set_title('Final Result (Sklearn)')
        ax_final.set_xlabel('Feature 1')
        ax_final.set_ylabel('Feature 2')
        ax_final.grid(True, alpha=0.3)
        ax_final.legend()
        
        plt.tight_layout()
        plt.suptitle('K-Means Clustering: Iterative Process', fontsize=16, y=0.98)
        
        # Save the figure
        if self.save_figures:
            filename = f'{self.output_dir}/kmeans_iterative_process.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        plt.show()
    
    def plot_convergence(self, X, iterations_data):
        """Plot convergence of centroids"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot centroid movement
        for cluster_id in range(self.n_clusters):
            centroid_history = [iter_data['centroids'][cluster_id] for iter_data in iterations_data]
            centroid_history = np.array(centroid_history)
            
            ax1.plot(centroid_history[:, 0], centroid_history[:, 1], 
                    'o-', color=self.colors[cluster_id], linewidth=2, markersize=8,
                    label=f'Centroid {cluster_id + 1}')
            
            # Mark start and end points
            ax1.scatter(centroid_history[0, 0], centroid_history[0, 1], 
                       c='black', s=100, marker='s', alpha=0.7)
            ax1.scatter(centroid_history[-1, 0], centroid_history[-1, 1], 
                       c='red', s=100, marker='*', alpha=0.7)
        
        ax1.set_title('Centroid Movement During Iterations')
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot within-cluster sum of squares
        wcss_history = []
        for iter_data in iterations_data:
            centroids = iter_data['centroids']
            labels = iter_data['labels']
            wcss = 0
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    wcss += np.sum((cluster_points - centroids[i]) ** 2)
            wcss_history.append(wcss)
        
        ax2.plot(range(1, len(wcss_history) + 1), wcss_history, 'bo-', linewidth=2, markersize=8)
        ax2.set_title('Within-Cluster Sum of Squares (WCSS) Convergence')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('WCSS')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        if self.save_figures:
            filename = f'{self.output_dir}/kmeans_convergence_analysis.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        plt.show()
    
    def create_animation_frames(self, X, iterations_data):
        """Create frames for potential GIF animation"""
        if not self.save_figures:
            return
            
        animation_dir = f'{self.output_dir}/animation_frames'
        os.makedirs(animation_dir, exist_ok=True)
        
        for i, iteration_data in enumerate(iterations_data):
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            centroids = iteration_data['centroids']
            labels = iteration_data['labels']
            iteration = iteration_data['iteration']
            
            # Plot data points
            for j in range(self.n_clusters):
                mask = labels == j
                ax.scatter(X[mask, 0], X[mask, 1], 
                          c=self.colors[j], alpha=0.7, s=50, 
                          label=f'Cluster {j+1}')
            
            # Plot centroids
            ax.scatter(centroids[:, 0], centroids[:, 1], 
                      c='black', marker='x', s=300, linewidths=4,
                      label='Centroids')
            
            ax.set_title(f'K-Means Clustering - Iteration {iteration + 1}', fontsize=16)
            ax.set_xlabel('Feature 1', fontsize=14)
            ax.set_ylabel('Feature 2', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            
            filename = f'{animation_dir}/frame_{i:03d}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Animation frames saved in: {animation_dir}")
    
    def run_demo(self):
        """Run the complete K-means clustering demo"""
        print("K-Means Clustering Demo")
        print("=" * 50)
        
        # Generate data
        print(f"Generating {self.n_samples} data points with {self.n_clusters} natural clusters...")
        X, y_true = self.generate_data()
        
        # Run manual K-means
        print("Running K-means algorithm...")
        iterations_data, final_labels, final_centroids = self.manual_kmeans(X)
        
        print(f"Algorithm converged after {len(iterations_data)} iterations")
        
        # Create and save visualizations
        print("Creating visualizations...")
        if self.save_figures:
            print(f"Figures will be saved to: {self.output_dir}/")
        
        self.visualize_clustering_process(X, iterations_data)
        self.plot_convergence(X, iterations_data)
        
        # Save individual iterations
        if self.save_figures:
            print("Saving individual iteration plots...")
            self.save_individual_iterations(X, iterations_data)
            
            print("Creating animation frames...")
            self.create_animation_frames(X, iterations_data)
        
        # Print final centroids
        print("\nFinal Centroids:")
        for i, centroid in enumerate(final_centroids):
            print(f"Cluster {i+1}: ({centroid[0]:.2f}, {centroid[1]:.2f})")
        
        # Calculate and print metrics
        from sklearn.metrics import silhouette_score, adjusted_rand_score
        
        silhouette_avg = silhouette_score(X, final_labels)
        ari_score = adjusted_rand_score(y_true, final_labels)
        
        print(f"\nClustering Metrics:")
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"Adjusted Rand Index: {ari_score:.3f}")
        
        if self.save_figures:
            print(f"\nAll figures saved to: {os.path.abspath(self.output_dir)}/")

def main():
    """Main function to run the demo"""
    # Create and run the demo with figure saving enabled
    demo = KMeansDemo(n_clusters=3, n_samples=300, random_state=42, save_figures=True)
    demo.run_demo()

if __name__ == "__main__":
    main()
