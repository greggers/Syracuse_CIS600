import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load a larger dataset: Breast Cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Evaluate tree performance with varying max_leaf_nodes
leaf_counts = list(range(2, 50))
accuracies = []

for leaf_count in leaf_counts:
    clf = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=leaf_count, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plot accuracy vs. number of leaves
plt.figure(figsize=(10, 6))
plt.plot(leaf_counts, accuracies, marker='o', linestyle='--', color='green')
plt.xlabel("Max Leaf Nodes")
plt.ylabel("Accuracy on Test Set")
plt.title("Pruning CART on Breast Cancer Dataset: Accuracy vs Max Leaf Nodes")
plt.grid(True)
plt.savefig("cart_pruning_plot.png")
plt.show()
