import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample production data
np.random.seed(0)
days = pd.date_range(start="2024-01-01", periods=100, freq="D")
# Generate data from two normal distributions
distribution1 = np.random.normal(loc=50, scale=5, size=15)  # N(50,5) with n=15
distribution2 = np.random.normal(loc=100, scale=20, size=85)  # N(100,20) with n=85
production_counts = np.concatenate([distribution1, distribution2])
# Shuffle the data to mix the two distributions
np.random.shuffle(production_counts)

# Create DataFrame for export to Excel
df = pd.DataFrame({
    "Date": days,
    "Production_Count": production_counts
})

# Save to Excel
excel_path = "./production_data.xlsx"
df.to_excel(excel_path, index=False)

# Python visualization (histogram)
plt.figure(figsize=(10, 6))
plt.hist(production_counts, bins=10, color='skyblue', edgecolor='black')
plt.title("Histogram of Production Counts")
plt.xlabel("Production Count")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()

# Save the histogram plot
plot_path = "./production_histogram.png"
plt.savefig(plot_path)

excel_path, plot_path
