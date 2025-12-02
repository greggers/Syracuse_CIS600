import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Generate synthetic customer data
np.random.seed(42)

# Features: Age, Income, Spending Score
n_customers = 200
ages = np.random.normal(40, 12, n_customers).clip(18, 70)
incomes = np.random.normal(60000, 15000, n_customers).clip(20000, 100000)
spending_scores = np.random.normal(50, 25, n_customers).clip(1, 100)

# Create a DataFrame
data = pd.DataFrame({
    'Age': ages,
    'Income': incomes,
    'SpendingScore': spending_scores
})

# Standardize features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_data)
data['KMeans_Cluster'] = kmeans_labels

# Hierarchical Clustering
linked = sch.linkage(scaled_data, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
sch.dendrogram(linked,
               orientation='top',
               labels=kmeans_labels,
               distance_sort='descending',
               show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Customer index')
plt.ylabel('Distance')
plt.show()

# Profile clusters
print("Cluster centers (KMeans):")
print(kmeans.cluster_centers_)

print("Customer data with cluster labels:")
print(data.head())
