"""
Bagging (Bootstrap Aggregating) Demo for Classification
This demonstrates how bagging works by creating multiple models on bootstrap samples
and combining their predictions through voting.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Generate a synthetic dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=10, 
    n_redundant=5, 
    n_classes=2, 
    random_state=42
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to implement manual bagging
def manual_bagging(X_train, y_train, X_test, n_estimators=10, sample_size=0.8):
    n_samples = int(len(X_train) * sample_size)
    predictions = np.zeros((X_test.shape[0], n_estimators))
    
    for i in range(n_estimators):
        # Bootstrap sampling
        indices = np.random.choice(len(X_train), n_samples, replace=True)
        X_bootstrap = X_train[indices]
        y_bootstrap = y_train[indices]
        
        # Train a base classifier
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X_bootstrap, y_bootstrap)
        
        # Store predictions
        predictions[:, i] = clf.predict(X_test)
    
    # Majority voting
    final_predictions = np.apply_along_axis(
        lambda x: np.bincount(x.astype('int')).argmax(), 
        axis=1, 
        arr=predictions
    )
    
    return final_predictions

# Compare manual bagging with scikit-learn's implementation
def compare_bagging_methods():
    # Manual bagging
    print("Implementing manual bagging...")
    manual_pred = manual_bagging(X_train, y_train, X_test, n_estimators=50)
    manual_acc = accuracy_score(y_test, manual_pred)
    
    # Scikit-learn bagging
    print("Using scikit-learn's BaggingClassifier...")
    bagging_clf = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=5),
        n_estimators=50,
        max_samples=0.8,
        bootstrap=True,
        random_state=42
    )
    bagging_clf.fit(X_train, y_train)
    sklearn_pred = bagging_clf.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_pred)
    
    # Single decision tree (for comparison)
    print("Training a single decision tree...")
    tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_clf.fit(X_train, y_train)
    tree_pred = tree_clf.predict(X_test)
    tree_acc = accuracy_score(y_test, tree_pred)
    
    print(f"\nAccuracy Comparison:")
    print(f"Single Decision Tree: {tree_acc:.4f}")
    print(f"Manual Bagging Implementation: {manual_acc:.4f}")
    print(f"Scikit-learn Bagging: {sklearn_acc:.4f}")
    
    return tree_acc, manual_acc, sklearn_acc

# Study the effect of the number of estimators
def study_n_estimators():
    n_estimators_range = [1, 5, 10, 20, 50, 100, 200]
    accuracies = []
    
    for n_est in n_estimators_range:
        bagging_clf = BaggingClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=5),
            n_estimators=n_est,
            max_samples=0.8,
            bootstrap=True,
            random_state=42
        )
        bagging_clf.fit(X_train, y_train)
        y_pred = bagging_clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, accuracies, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Effect of Number of Estimators on Bagging Performance')
    plt.xscale('log')
    plt.grid(True)
    plt.savefig('bagging_n_estimators.png')
    plt.close()
    
    print("\nEffect of number of estimators saved to 'bagging_n_estimators.png'")

if __name__ == "__main__":
    print("Bagging Demo for Classification")
    print("=" * 40)
    
    # Compare different bagging implementations
    tree_acc, manual_acc, sklearn_acc = compare_bagging_methods()
    
    # Study the effect of the number of estimators
    study_n_estimators()
    
    # Visualize accuracy comparison
    methods = ['Single Tree', 'Manual Bagging', 'Scikit-learn Bagging']
    accuracies = [tree_acc, manual_acc, sklearn_acc]
    
    plt.figure(figsize=(10, 6))
    plt.bar(methods, accuracies, color=['lightblue', 'lightgreen', 'coral'])
    plt.ylabel('Accuracy')
    plt.title('Comparison of Classification Methods')
    plt.ylim(0.7, 1.0)  # Adjust as needed
    
    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    plt.savefig('bagging_comparison.png')
    plt.close()
    
    print("\nAccuracy comparison saved to 'bagging_comparison.png'")
    print("\nBagging demo completed!")