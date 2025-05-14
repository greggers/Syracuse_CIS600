"""
Boosting Demo for Classification
This demonstrates how boosting algorithms like AdaBoost, Gradient Boosting, and XGBoost
work for classification tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Generate a synthetic dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=10, 
    n_redundant=5, 
    n_classes=2, 
    weights=[0.7, 0.3],  # Make it slightly imbalanced
    random_state=42
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to implement a simplified version of AdaBoost
def manual_adaboost(X_train, y_train, X_test, n_estimators=50, learning_rate=1.0):
    n_samples = X_train.shape[0]
    # Initialize weights uniformly
    sample_weights = np.ones(n_samples) / n_samples
    
    # Convert y to {-1, 1} for easier calculation
    y_train_mod = np.where(y_train == 0, -1, 1)
    
    # Initialize predictions
    predictions = np.zeros(X_test.shape[0])
    
    # List to store estimators and their weights
    estimators = []
    alphas = []
    
    for i in range(n_estimators):
        # Train a weak learner with current weights
        clf = DecisionTreeClassifier(max_depth=1)  # Stump
        clf.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Make predictions
        y_pred = clf.predict(X_train)
        y_pred_mod = np.where(y_pred == 0, -1, 1)
        
        # Calculate error
        err = np.sum(sample_weights * (y_pred_mod != y_train_mod)) / np.sum(sample_weights)
        
        # Calculate alpha (estimator weight)
        alpha = learning_rate * 0.5 * np.log((1 - err) / max(err, 1e-10))
        
        # Update sample weights
        sample_weights *= np.exp(-alpha * y_train_mod * y_pred_mod)
        sample_weights /= np.sum(sample_weights)  # Normalize
        
        # Store the estimator and its weight
        estimators.append(clf)
        alphas.append(alpha)
        
        # Update predictions
        predictions += alpha * np.where(clf.predict(X_test) == 0, -1, 1)
    
    # Convert back to 0/1 predictions
    final_predictions = np.where(predictions < 0, 0, 1)
    
    return final_predictions, estimators, alphas

# Compare different boosting algorithms
def compare_boosting_algorithms():
    results = {}
    
    # Base classifier (Decision Tree)
    print("Training a single decision tree...")
    tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_clf.fit(X_train_scaled, y_train)
    tree_pred = tree_clf.predict(X_test_scaled)
    tree_acc = accuracy_score(y_test, tree_pred)
    results['Decision Tree'] = tree_acc
    
    # Manual AdaBoost implementation
    print("Implementing manual AdaBoost...")
    manual_pred, _, _ = manual_adaboost(X_train_scaled, y_train, X_test_scaled, n_estimators=50)
    manual_acc = accuracy_score(y_test, manual_pred)
    results['Manual AdaBoost'] = manual_acc
    
    # Scikit-learn AdaBoost
    print("Using scikit-learn's AdaBoostClassifier...")
    ada_clf = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    ada_clf.fit(X_train_scaled, y_train)
    ada_pred = ada_clf.predict(X_test_scaled)
    ada_acc = accuracy_score(y_test, ada_pred)
    results['AdaBoost'] = ada_acc
    
    # Gradient Boosting
    print("Using GradientBoostingClassifier...")
    gb_clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_clf.fit(X_train_scaled, y_train)
    gb_pred = gb_clf.predict(X_test_scaled)
    gb_acc = accuracy_score(y_test, gb_pred)
    results['Gradient Boosting'] = gb_acc
    
    # XGBoost
    print("Using XGBoost...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    xgb_clf.fit(X_train_scaled, y_train)
    xgb_pred = xgb_clf.predict(X_test_scaled)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    results['XGBoost'] = xgb_acc
    
    print(f"\nAccuracy Comparison:")
    for method, acc in results.items():
        print(f"{method}: {acc:.4f}")
    
    # Print detailed report for the best method
    best_method = max(results, key=results.get)
    print(f"\nDetailed report for {best_method}:")
    if best_method == 'XGBoost':
        print(classification_report(y_test, xgb_pred))
    elif best_method == 'Gradient Boosting':
        print(classification_report(y_test, gb_pred))
    elif best_method == 'AdaBoost':
        print(classification_report(y_test, ada_pred))
    elif best_method == 'Manual AdaBoost':
        print(classification_report(y_test, manual_pred))
    else:
        print(classification_report(y_test, tree_pred))
    
    return results

# Study the effect of the number of estimators for AdaBoost
def study_adaboost_n_estimators():
    n_estimators_range = [1, 5, 10, 20, 50, 100, 200]
    accuracies = []
    
    for n_est in n_estimators_range:
        ada_clf = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=n_est,
            learning_rate=1.0,
            random_state=42
        )
        ada_clf.fit(X_train_scaled, y_train)
        y_pred = ada_clf.predict(X_test_scaled)
        accuracies.append(accuracy_score(y_test, y_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, accuracies, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Effect of Number of Estimators on AdaBoost Performance')
    plt.xscale('log')
    plt.grid(True)
    plt.savefig('adaboost_n_estimators.png')
    plt.close()
    
    print("\nEffect of number of estimators saved to 'adaboost_n_estimators.png'")

# Study the effect of learning rate for Gradient Boosting
def study_gb_learning_rate():
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
    gb_accuracies = []
    xgb_accuracies = []
    
    for lr in learning_rates:
        # Gradient Boosting
        gb_clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=lr,
            max_depth=3,
            random_state=42
        )
        gb_clf.fit(X_train_scaled, y_train)
        gb_pred = gb_clf.predict(X_test_scaled)
        gb_accuracies.append(accuracy_score(y_test, gb_pred))
        
        # XGBoost
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=lr,
            max_depth=3,
            random_state=42
        )
        xgb_clf.fit(X_train_scaled, y_train)
        xgb_pred = xgb_clf.predict(X_test_scaled)
        xgb_accuracies.append(accuracy_score(y_test, xgb_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, gb_accuracies, marker='o', label='Gradient Boosting')
    plt.plot(learning_rates, xgb_accuracies, marker='s', label='XGBoost')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Effect of Learning Rate on Boosting Performance')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('boosting_learning_rate.png')
    plt.close()
    
    print("\nEffect of learning rate saved to 'boosting_learning_rate.png'")

# Visualize the decision boundaries of different boosting algorithms
def visualize_decision_boundaries():
    # Create a simpler 2D dataset for visualization
    X_2d, y_2d = make_classification(
        n_samples=1000, 
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42
    )
    
    X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
        X_2d, y_2d, test_size=0.3, random_state=42
    )
    
    # Train different models
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
        'AdaBoost': AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=50,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    }
    
    # Fit all models
    for name, model in models.items():
        model.fit(X_train_2d, y_train_2d)
    
    # Create a meshgrid for plotting decision boundaries
    h = 0.02  # Step size
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Plot decision boundaries
    plt.figure(figsize=(15, 10))
    
    for i, (name, model) in enumerate(models.items()):
        plt.subplot(2, 2, i + 1)
        
        # Plot decision boundary
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
        
        # Plot training points
        plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train_2d, 
                   edgecolors='k', marker='o', s=50, cmap=plt.cm.RdBu)
        
        # Plot test points
        plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test_2d,
                   edgecolors='k', marker='^', s=50, cmap=plt.cm.RdBu, alpha=0.6)
        
        plt.title(f"{name} - Accuracy: {accuracy_score(y_test_2d, model.predict(X_test_2d)):.4f}")
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('boosting_decision_boundaries.png')
    plt.close()
    
    print("\nDecision boundaries visualization saved to 'boosting_decision_boundaries.png'")

if __name__ == "__main__":
    print("Boosting Demo for Classification")
    print("=" * 40)
    
    # Compare different boosting algorithms
    results = compare_boosting_algorithms()
    
    # Study the effect of the number of estimators for AdaBoost
    study_adaboost_n_estimators()
    
    # Study the effect of learning rate for Gradient Boosting
    study_gb_learning_rate()
    
    # Visualize decision boundaries
    visualize_decision_boundaries()
    
    # Visualize accuracy comparison
    methods = list(results.keys())
    accuracies = list(results.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(methods, accuracies, color=['lightblue', 'lightgreen', 'coral', 'gold', 'orchid'])
    plt.ylabel('Accuracy')
    plt.title('Comparison of Boosting Methods')
    plt.ylim(0.7, 1.0)  # Adjust as needed
    plt.xticks(rotation=45)
    
    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('boosting_comparison.png')
    plt.close()
    
    print("\nAccuracy comparison saved to 'boosting_comparison.png'")
    print("\nBoosting demo completed!")
    
    # Additional explanation
    print("\nKey Insights about Boosting Algorithms:")
    print("1. AdaBoost focuses on misclassified samples by adjusting sample weights.")
    print("2. Gradient Boosting builds trees sequentially to correct errors of previous trees.")
    print("3. XGBoost is an optimized implementation of gradient boosting with regularization.")
    print("4. Boosting typically outperforms bagging for bias reduction but may be more prone to overfitting.")
    print("5. Learning rate controls the contribution of each tree, with smaller values often yielding better generalization.")