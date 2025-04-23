import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import seaborn as sns

# Read the Excel file
file_path = './production_data.xlsx'
df = pd.read_excel(file_path)

# Assuming Column A is the date and Column B is the production counts
# Rename columns for clarity
df.columns = ['Date', 'Production'] if len(df.columns) >= 2 else df.columns

# 1. Create a histogram of production counts
plt.figure(figsize=(10, 6))
plt.hist(df['Production'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Histogram of Production Counts')
plt.xlabel('Production Count')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig('production_histogram.png')
plt.show()

# 2. Create a Gaussian Mixture Model with K=2
# Reshape data for GMM
X = df['Production'].values.reshape(-1, 1)

# Fit GMM
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X)

# Generate a range of x values for plotting the GMM
x = np.linspace(df['Production'].min() - 1, df['Production'].max() + 1, 1000).reshape(-1, 1)
logprob = gmm.score_samples(x)
responsibilities = gmm.predict_proba(x)

# Plot histogram with GMM overlay
plt.figure(figsize=(10, 6))
plt.hist(df['Production'], bins=20, density=True, alpha=0.5, color='skyblue', edgecolor='black')

# Plot the GMM components
pdf = np.exp(logprob)
plt.plot(x, pdf, '-k', label='Mixture PDF')

# Plot individual Gaussian components
for i in range(2):
    pdf_component = responsibilities[:, i] * pdf
    plt.plot(x, pdf_component, '--', label=f'Component {i+1}')

plt.title('Gaussian Mixture Model (K=2) of Production Counts')
plt.xlabel('Production Count')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('production_gmm.png')
plt.show()

# 3. Convert Excel to CSV for Neo4J
# Add an ID column for Neo4J relationships if needed
df['ID'] = range(1, len(df) + 1)

# Reorder columns to have ID first
df = df[['ID', 'Date', 'Production']]

# Save to CSV
csv_path = './production_data.csv'
df.to_csv(csv_path, index=False)

print(f"Data successfully converted to CSV and saved at {csv_path}")
print("Histogram and GMM plots have been created and saved.")

# Display summary statistics
print("\nSummary Statistics of Production Data:")
print(df['Production'].describe())

# Show the first few rows of the CSV data
print("\nFirst few rows of the CSV data:")
print(df.head())
