import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# 1. Load and prepare dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Create 80%, 15%, 5% split for train, validate, test
# First split: 80% train, 20% remaining
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Second split: divide the 20% into 15% validate and 5% test (which is a 75%/25% split of the remaining data)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Verify the proportions
print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]:.1%})")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/X.shape[0]:.1%})")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]:.1%})")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 2. Resubstitution demonstration
def demonstrate_resubstitution():
    print("\nDemonstrating Resubstitution Method")
    model = RandomForestClassifier(random_state=42)
    
    # Train on all training data
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on training data (resubstitution)
    train_pred = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    
    # Evaluate on validation data
    val_pred = model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_pred)
    
    # Evaluate on test data
    test_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"Training accuracy (resubstitution): {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Difference (train vs. val - potential overfitting): {train_acc - val_acc:.4f}")
    
    return model, train_acc, val_acc, test_acc

# 3. Cross-validation demonstration
def demonstrate_cross_validation():
    print("\nDemonstrating K-Fold Cross-Validation")
    model = RandomForestClassifier(random_state=42)
    
    # Perform 5-fold cross-validation on training data
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")
    print(f"Standard deviation: {np.std(cv_scores):.4f}")
    
    # Train on full training set and evaluate on validation set
    model.fit(X_train_scaled, y_train)
    val_acc = model.score(X_val_scaled, y_val)
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Evaluate on test set
    test_acc = model.score(X_test_scaled, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    return cv_scores, val_acc, test_acc

# 4. Hyperparameter tuning with GridSearchCV
def demonstrate_hyperparameter_tuning():
    print("\nDemonstrating Hyperparameter Tuning with GridSearchCV")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    # Create GridSearchCV object
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Fit GridSearchCV on training data
    grid_search.fit(X_train_scaled, y_train)
    
    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate best model on validation set
    best_model = grid_search.best_estimator_
    val_acc = best_model.score(X_val_scaled, y_val)
    print(f"Validation accuracy with best model: {val_acc:.4f}")
    
    # Final evaluation on test set
    test_acc = best_model.score(X_test_scaled, y_test)
    print(f"Test accuracy with best model: {test_acc:.4f}")
    
    return grid_search, val_acc, test_acc

# 5. Manual hyperparameter tuning with validation set
def manual_hyperparameter_tuning():
    print("\nDemonstrating Manual Hyperparameter Tuning with Validation Set")
    
    # Define hyperparameters to try
    n_estimators_list = [50, 100, 200]
    max_depth_list = [None, 10, 20, 30]
    
    best_val_acc = 0
    best_params = {}
    
    # Manually try different combinations
    for n_est in n_estimators_list:
        for depth in max_depth_list:
            # Create and train model
            model = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=depth,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate on validation set
            val_acc = model.score(X_val_scaled, y_val)
            
            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {'n_estimators': n_est, 'max_depth': depth}
                best_model = model
    
    print(f"Best parameters from manual tuning: {best_params}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Evaluate best model on test set
    test_acc = best_model.score(X_test_scaled, y_test)
    print(f"Test accuracy with best model: {test_acc:.4f}")
    
    return best_model, best_params, best_val_acc, test_acc

# 6. Visualize results
def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None):
    from sklearn.model_selection import learning_curve
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy'
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    return plt

# Run demonstrations
model, train_acc, val_acc, resub_test_acc = demonstrate_resubstitution()
cv_scores, cv_val_acc, cv_test_acc = demonstrate_cross_validation()
grid_search, gs_val_acc, gs_test_acc = demonstrate_hyperparameter_tuning()
best_manual_model, best_manual_params, manual_val_acc, manual_test_acc = manual_hyperparameter_tuning()

# Plot learning curve for the best model
plot_learning_curve(
    grid_search.best_estimator_,
    "Learning Curve (Random Forest)",
    X_train_scaled, y_train,
    cv=5
)
plt.savefig('learning_curve.png')
plt.show()

# Compare different evaluation methods
methods = ['Resubstitution', '5-Fold CV', 'GridSearchCV', 'Manual Tuning']
val_accuracies = [val_acc, cv_val_acc, gs_val_acc, manual_val_acc]
test_accuracies = [resub_test_acc, cv_test_acc, gs_test_acc, manual_test_acc]

plt.figure(figsize=(12, 6))
x = np.arange(len(methods))
width = 0.35

plt.bar(x - width/2, val_accuracies, width, label='Validation Accuracy')
plt.bar(x + width/2, test_accuracies, width, label='Test Accuracy')

plt.xlabel('Method')
plt.ylabel('Accuracy')
plt.title('Validation and Test Accuracy by Evaluation Method')
plt.xticks(x, methods)
plt.legend()
plt.ylim(0.9, 1.0)  # Adjust as needed
plt.savefig('evaluation_comparison.png')
plt.show()

# Visualize overfitting by comparing train, validation, and test accuracies
plt.figure(figsize=(10, 6))
methods = ['Resubstitution', '5-Fold CV', 'GridSearchCV', 'Manual Tuning']
train_accs = [train_acc, np.mean(cv_scores), grid_search.best_score_, train_acc]  # Using train_acc as proxy for manual

x = np.arange(len(methods))
width = 0.25

plt.bar(x - width, train_accs, width, label='Training Accuracy')
plt.bar(x, val_accuracies, width, label='Validation Accuracy')
plt.bar(x + width, test_accuracies, width, label='Test Accuracy')

plt.xlabel('Method')
plt.ylabel('Accuracy')
plt.title('Training, Validation, and Test Accuracy Comparison')
plt.xticks(x, methods)
plt.legend()
plt.ylim(0.9, 1.0)  # Adjust as needed
plt.savefig('overfitting_comparison.png')
plt.show()
