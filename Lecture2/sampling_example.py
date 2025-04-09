import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import norm, multivariate_normal
import matplotlib.gridspec as gridspec

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 100

# Create figure
plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

# ================ UNIFORM RANDOM SAMPLING ================

# 1D Uniform Random Sampling
ax1 = plt.subplot(gs[0, 0])
samples_1d_uniform = np.random.uniform(0, 1, n_samples)
ax1.scatter(samples_1d_uniform, np.zeros_like(samples_1d_uniform), 
           color='blue', alpha=0.7, s=30)
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.1, 0.1)
ax1.set_title(f'1D Uniform Random Sampling ({n_samples} samples)')
ax1.set_xlabel('X')
ax1.set_yticks([])
ax1.grid(True, alpha=0.3)

# 2D Uniform Random Sampling
ax2 = plt.subplot(gs[0, 1])
samples_2d_uniform = np.random.uniform(0, 1, (n_samples, 2))
ax2.scatter(samples_2d_uniform[:, 0], samples_2d_uniform[:, 1], 
           color='blue', alpha=0.7, s=30)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title(f'2D Uniform Random Sampling ({n_samples} samples)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.grid(True, alpha=0.3)

# Calculate coverage metrics for uniform sampling
# For 1D: average distance between adjacent points
sorted_1d = np.sort(samples_1d_uniform)
avg_dist_1d = np.mean(np.diff(sorted_1d))
max_gap_1d = np.max(np.diff(sorted_1d))

# For 2D: average distance to nearest neighbor
from scipy.spatial import cKDTree
tree = cKDTree(samples_2d_uniform)
distances, _ = tree.query(samples_2d_uniform, k=2)
avg_nn_dist = np.mean(distances[:, 1])  # distances[:,0] is distance to self (0)

# ================ MONTE CARLO SAMPLING ================

# 1D Monte Carlo Sampling with importance sampling
# Using a beta distribution to concentrate points in areas of interest
ax3 = plt.subplot(gs[1, 0])
# Beta distribution parameters to concentrate points near 0.3 and 0.7
samples_1d_mc = np.random.beta(2, 2, n_samples)
ax3.scatter(samples_1d_mc, np.zeros_like(samples_1d_mc), 
           color='green', alpha=0.7, s=30)
ax3.set_xlim(0, 1)
ax3.set_ylim(-0.1, 0.1)
ax3.set_title(f'1D Monte Carlo Sampling - Beta(2,2) ({n_samples} samples)')
ax3.set_xlabel('X')
ax3.set_yticks([])
ax3.grid(True, alpha=0.3)

# Plot the PDF of Beta(2,2)
x = np.linspace(0, 1, 1000)
from scipy.stats import beta
pdf_beta = beta.pdf(x, 2, 2)
ax3_twin = ax3.twinx()
ax3_twin.plot(x, pdf_beta, 'r-', lw=2)
ax3_twin.set_ylabel('PDF of Beta(2,2)', color='r')
ax3_twin.tick_params(axis='y', colors='r')

# 2D Monte Carlo Sampling with importance sampling
# Using a mixture of Gaussians to concentrate points in areas of interest
ax4 = plt.subplot(gs[1, 1])

# Create a mixture of two Gaussians
def mixture_of_gaussians(n_samples):
    # First Gaussian centered at (0.3, 0.3)
    samples1 = np.random.multivariate_normal(
        mean=[0.3, 0.3], 
        cov=[[0.02, 0], [0, 0.02]], 
        size=n_samples//2
    )
    
    # Second Gaussian centered at (0.7, 0.7)
    samples2 = np.random.multivariate_normal(
        mean=[0.7, 0.7], 
        cov=[[0.02, 0], [0, 0.02]], 
        size=n_samples//2
    )
    
    # Combine samples
    return np.vstack([samples1, samples2])

samples_2d_mc = mixture_of_gaussians(n_samples)
# Clip to ensure all points are in [0,1]×[0,1]
samples_2d_mc = np.clip(samples_2d_mc, 0, 1)

ax4.scatter(samples_2d_mc[:, 0], samples_2d_mc[:, 1], 
           color='green', alpha=0.7, s=30)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_title(f'2D Monte Carlo Sampling - Mixture of Gaussians ({n_samples} samples)')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.grid(True, alpha=0.3)

# ================ COMPARISON OF FUNCTION ESTIMATION ================

# Define a test function: f(x) = sin(2πx) for 1D, f(x,y) = sin(2πx)cos(2πy) for 2D
def f_1d(x):
    return np.sin(2 * np.pi * x)

def f_2d(x, y):
    return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

# Compute true integrals (analytically)
true_integral_1d = 0  # ∫sin(2πx)dx from 0 to 1 = 0
true_integral_2d = 0  # ∫∫sin(2πx)cos(2πy)dxdy from [0,0] to [1,1] = 0

# Estimate integrals using samples
estimate_uniform_1d = np.mean(f_1d(samples_1d_uniform))
estimate_mc_1d = np.mean(f_1d(samples_1d_mc))

estimate_uniform_2d = np.mean(f_2d(samples_2d_uniform[:, 0], samples_2d_uniform[:, 1]))
estimate_mc_2d = np.mean(f_2d(samples_2d_mc[:, 0], samples_2d_mc[:, 1]))

# Plot the test functions and estimation errors
ax5 = plt.subplot(gs[2, 0])
x_fine = np.linspace(0, 1, 1000)
ax5.plot(x_fine, f_1d(x_fine), 'k-', lw=2, label='f(x) = sin(2πx)')
ax5.scatter(samples_1d_uniform, f_1d(samples_1d_uniform), 
           color='blue', alpha=0.5, s=30, label='Uniform')
ax5.scatter(samples_1d_mc, f_1d(samples_1d_mc), 
           color='green', alpha=0.5, s=30, label='Monte Carlo')
ax5.set_xlim(0, 1)
ax5.set_title('1D Function Estimation')
ax5.set_xlabel('X')
ax5.set_ylabel('f(x)')
ax5.grid(True, alpha=0.3)
ax5.legend()

# Add text with estimation errors
ax5.text(0.05, 0.95, 
         f"True integral = {true_integral_1d}\n"
         f"Uniform estimate = {estimate_uniform_1d:.4f}\n"
         f"MC estimate = {estimate_mc_1d:.4f}",
         transform=ax5.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# 2D function visualization
ax6 = plt.subplot(gs[2, 1])
x_grid, y_grid = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
z_grid = f_2d(x_grid, y_grid)
contour = ax6.contourf(x_grid, y_grid, z_grid, cmap='viridis', levels=50)
plt.colorbar(contour, ax=ax6, label='f(x,y)')

# Plot the sample points
ax6.scatter(samples_2d_uniform[:, 0], samples_2d_uniform[:, 1], 
           color='blue', alpha=0.7, s=30, label='Uniform')
ax6.scatter(samples_2d_mc[:, 0], samples_2d_mc[:, 1], 
           color='green', alpha=0.7, s=30, label='Monte Carlo')
ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)
ax6.set_title('2D Function Estimation: f(x,y) = sin(2πx)cos(2πy)')
ax6.set_xlabel('X')
ax6.set_ylabel('Y')
ax6.legend()

# Add text with estimation errors
ax6.text(0.05, 0.95, 
         f"True integral = {true_integral_2d}\n"
         f"Uniform estimate = {estimate_uniform_2d:.4f}\n"
         f"MC estimate = {estimate_mc_2d:.4f}",
         transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.suptitle('Comparison of Uniform Random Sampling vs. Monte Carlo Sampling', fontsize=16, y=1.02)
plt.savefig('sampling_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
