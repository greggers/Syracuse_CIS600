"""
Random Forests Demo for Classification
This demonstrates how Random Forests work and compares them with individual decision trees
and other ensemble methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.inspection import permutation_importance
import seaborn as sns
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

def load_data(use_real_data=True):
    """Load either synthetic or real dataset for classification"""
    if use_real_data:
        # Load breast cancer dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names
        print(f"Loaded breast cancer dataset with {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: {np.bincount(y)}")
    else:
        # Generate a synthetic dataset
        X, y = make_classification(
            n_samples=1000, 
            n_features=20, 
            n_informative=10, 
            n_redundant=5, 
            n_classes=2, 
            random_state=42
        )
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        print(f"Generated synthetic dataset with {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test, feature_names

def compare_models(X_train, X_test, y_train, y_test):
    """Compare different tree-based models"""
    results = {}
    
    # Single Decision Tree
    print("Training a single decision tree...")
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    tree_pred = tree_clf.predict(X_test)
    tree_acc = accuracy_score(y_test, tree_pred)
    results['Decision Tree'] = tree_acc
    
    # Random Forest with different numbers of trees
    for n_trees in [10, 50, 100]:
        print(f"Training Random Forest with {n_trees} trees...")
        rf_clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        rf_clf.fit(X_train, y_train)
        rf_pred = rf_clf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        results[f'Random Forest ({n_trees} trees)'] = rf_acc
    
    # Extra Trees (Extremely Randomized Trees)
    print("Training Extra Trees classifier...")
    et_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
    et_clf.fit(X_train, y_train)
    et_pred = et_clf.predict(X_test)
    et_acc = accuracy_score(y_test, et_pred)
    results['Extra Trees'] = et_acc
    
    # Print results
    print("\nAccuracy Comparison:")
    for model, acc in results.items():
        print(f"{model}: {acc:.4f}")
    
    # Get the best model for further analysis
    best_model_name = max(results, key=results.get)
    print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]:.4f}")
    
    # Get the best Random Forest model for further analysis
    rf_models = {k: v for k, v in results.items() if 'Random Forest' in k}
    best_rf_name = max(rf_models, key=rf_models.get)
    
    # Train the best Random Forest model again
    n_trees = int(best_rf_name.split('(')[1].split(' ')[0])
    best_rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    best_rf.fit(X_train, y_train)
    
    return results, best_rf

def analyze_feature_importance(model, X_train, y_train, X_test, feature_names):
    """Analyze feature importance in Random Forest model"""
    print("\nAnalyzing feature importance...")
    
    # Get feature importance from the model
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print feature ranking
    print("Feature ranking:")
    for i in range(min(10, len(feature_names))):
        print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")
    
    # Plot feature importances
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(min(20, len(feature_names))), 
            importances[indices[:20]],
            align="center")
    plt.xticks(range(min(20, len(feature_names))), 
               [feature_names[i] for i in indices[:20]], 
               rotation=90)
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png')
    plt.close()
    
    # Compute permutation importance
    print("\nComputing permutation importance (this may take a moment)...")
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    
    # Sort features by permutation importance
    perm_indices = perm_importance.importances_mean.argsort()[::-1]
    
    # Plot permutation importances
    plt.figure(figsize=(12, 6))
    plt.title("Permutation Importances")
    plt.bar(range(min(20, len(feature_names))), 
            perm_importance.importances_mean[perm_indices[:20]],
            yerr=perm_importance.importances_std[perm_indices[:20]],
            align="center")
    plt.xticks(range(min(20, len(feature_names))), 
               [feature_names[i] for i in perm_indices[:20]], 
               rotation=90)
    plt.tight_layout()
    plt.savefig('rf_permutation_importance.png')
    plt.close()
    
    print("Feature importance analysis completed and saved to 'rf_feature_importance.png' and 'rf_permutation_importance.png'")

def analyze_oob_error(X_train, y_train):
    """Analyze Out-of-Bag error as a function of the number of trees"""
    print("\nAnalyzing Out-of-Bag error...")
    
    n_estimators = list(range(1, 151, 10))
    oob_errors = []
    
    for n_est in n_estimators:
        rf = RandomForestClassifier(n_estimators=n_est, oob_score=True, random_state=42)
        rf.fit(X_train, y_train)
        oob_errors.append(1 - rf.oob_score_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators, oob_errors, 'bo-')
    plt.xlabel('Number of Trees')
    plt.ylabel('OOB Error Rate')
    plt.title('Random Forest: OOB Error Rate vs Number of Trees')
    plt.grid(True)
    plt.savefig('rf_oob_error.png')
    plt.close()
    
    print("OOB error analysis completed and saved to 'rf_oob_error.png'")

def analyze_max_features(X_train, X_test, y_train, y_test):
    """Analyze the effect of max_features parameter"""
    print("\nAnalyzing the effect of max_features parameter...")
    
    # Different values for max_features
    if X_train.shape[1] > 10:
        max_features_options = [1, 2, 'sqrt', 'log2', 0.3, 0.5, 0.7, 1.0]
    else:
        max_features_options = [1, 2, 'sqrt', 'log2', 0.5, 1.0]
    
    accuracies = []
    
    for max_feat in max_features_options:
        rf = RandomForestClassifier(n_estimators=100, max_features=max_feat, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    
    plt.figure(figsize=(10, 6))
    plt.bar([str(mf) for mf in max_features_options], accuracies)
    plt.xlabel('max_features')
    plt.ylabel('Accuracy')
    plt.title('Random Forest: Accuracy vs max_features')
    plt.ylim(min(accuracies) - 0.05, max(accuracies) + 0.05)
    
    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('rf_max_features.png')
    plt.close()
    
    print("max_features analysis completed and saved to 'rf_max_features.png'")

def visualize_tree_depth_effect(X_train, X_test, y_train, y_test):
    """Visualize the effect of maximum tree depth"""
    print("\nAnalyzing the effect of maximum tree depth...")
    
    max_depths = [1, 2, 3, 5, 10, 20, None]
    dt_scores = []
    rf_scores = []
    
    for depth in max_depths:
        # Decision Tree
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        dt_scores.append(accuracy_score(y_test, dt.predict(X_test)))
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
        rf.fit(X_train, y_train)
        rf_scores.append(accuracy_score(y_test, rf.predict(X_test)))
    
    # Convert None to "None" for plotting
    max_depths_str = [str(d) for d in max_depths]
    
    plt.figure(figsize=(10, 6))
    plt.plot(max_depths_str, dt_scores, 'o-', label='Decision Tree')
    plt.plot(max_depths_str, rf_scores, 's-', label='Random Forest')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Accuracy')
    plt.title('Effect of Tree Depth on Model Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('rf_tree_depth.png')
    plt.close()
    
    print("Tree depth analysis completed and saved to 'rf_tree_depth.png'")

def visualize_decision_boundaries():
    """Visualize decision boundaries of Random Forest vs Decision Tree"""
    print("\nVisualizing decision boundaries...")
    
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
    
    # Train models
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest (10 trees)': RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42),
        'Random Forest (100 trees)': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42)
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
    plt.savefig('rf_decision_boundaries.png')
    plt.close()
    
    print("Decision boundaries visualization saved to 'rf_decision_boundaries.png'")

def plot_learning_curve(X, y):
    """Plot learning curves for Random Forest vs Decision Tree"""
    print("\nPlotting learning curves...")
    
        # Define models
    tree = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Define training sizes
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Create figure
    plt.figure(figsize=(12, 5))
    
    # Plot learning curve for Decision Tree
    plt.subplot(1, 2, 1)
    train_sizes, train_scores, test_scores = learning_curve(
        tree, X, y, train_sizes=train_sizes, cv=5, scoring='accuracy', n_jobs=-1)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    plt.title('Decision Tree Learning Curve')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Plot learning curve for Random Forest
    plt.subplot(1, 2, 2)
    train_sizes, train_scores, test_scores = learning_curve(
        rf, X, y, train_sizes=train_sizes, cv=5, scoring='accuracy', n_jobs=-1)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    plt.title('Random Forest Learning Curve')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('rf_learning_curves.png')
    plt.close()
    
    print("Learning curves saved to 'rf_learning_curves.png'")

def demonstrate_bootstrap_sampling():
    """Demonstrate how bootstrap sampling works in Random Forests"""
    print("\nDemonstrating bootstrap sampling...")
    
    # Create a small dataset for demonstration
    np.random.seed(42)
    X_small = np.random.rand(10, 2)
    y_small = np.random.randint(0, 2, 10)
    
    # Original dataset
    print("Original dataset (10 samples):")
    for i in range(len(X_small)):
        print(f"Sample {i}: {X_small[i]}, class: {y_small[i]}")
    
    # Create bootstrap samples
    n_bootstraps = 3
    bootstrap_samples = []
    
    for i in range(n_bootstraps):
        indices = np.random.choice(len(X_small), len(X_small), replace=True)
        bootstrap_samples.append(indices)
    
    # Print bootstrap samples
    for i, sample in enumerate(bootstrap_samples):
        print(f"\nBootstrap sample {i+1}:")
        for j, idx in enumerate(sample):
            print(f"  Position {j}: Original sample {idx}: {X_small[idx]}, class: {y_small[idx]}")
        
        # Count occurrences
        unique, counts = np.unique(sample, return_counts=True)
        print("\n  Sample counts:")
        for u, c in zip(unique, counts):
            print(f"  Original sample {u} appears {c} times")
        
        # Check which original samples are not in this bootstrap
        out_of_bag = np.setdiff1d(np.arange(len(X_small)), sample)
        print(f"\n  Out-of-bag samples: {out_of_bag}")
    
    print("\nThis demonstrates how Random Forests use bootstrap sampling to create diverse trees.")
    print("Each tree is trained on a different bootstrap sample of the original data.")
    print("Out-of-bag samples can be used for validation without needing a separate test set.")

if __name__ == "__main__":
    print("Random Forests Demo for Classification")
    print("=" * 40)
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_data(use_real_data=True)
    
    # Compare different models
    results, best_rf = compare_models(X_train, X_test, y_train, y_test)
    
    # Analyze feature importance
    analyze_feature_importance(best_rf, X_train, y_train, X_test, feature_names)
    
    # Analyze OOB error
    analyze_oob_error(X_train, y_train)
    
    # Analyze max_features parameter
    analyze_max_features(X_train, X_test, y_train, y_test)
    
    # Visualize tree depth effect
    visualize_tree_depth_effect(X_train, X_test, y_train, y_test)
    
    # Visualize decision boundaries
    visualize_decision_boundaries()
    
    # Plot learning curves
    plot_learning_curve(np.vstack((X_train, X_test)), np.concatenate((y_train, y_test)))
    
    # Demonstrate bootstrap sampling
    demonstrate_bootstrap_sampling()
    
    # Visualize model comparison
    methods = list(results.keys())
    accuracies = list(results.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(methods, accuracies, color=['lightblue', 'lightgreen', 'coral', 'gold', 'orchid'])
    plt.ylabel('Accuracy')
    plt.title('Comparison of Tree-based Methods')
    plt.ylim(0.8, 1.0)  # Adjust as needed
    plt.xticks(rotation=45)
    
    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('rf_comparison.png')
    plt.close()
    
    print("\nAccuracy comparison saved to 'rf_comparison.png'")
    print("\nRandom Forests demo completed!")
    
    # Additional explanation
    print("\nKey Insights about Random Forests:")
    print("1. Random Forests reduce overfitting by averaging multiple decision trees.")
    print("2. Each tree is trained on a bootstrap sample of the data (bagging).")
    print("3. Random feature selection at each split increases diversity among trees.")
    print("4. Out-of-bag samples provide a built-in validation set for each tree.")
    print("5. Random Forests typically outperform single decision trees in accuracy and robustness.")
    print("6. They provide feature importance measures to help with feature selection and interpretation.")
    print("7. Extra Trees (Extremely Randomized Trees) add even more randomization by using random thresholds.")
