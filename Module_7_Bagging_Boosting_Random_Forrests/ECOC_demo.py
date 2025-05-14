"""
Error-Correcting Output Coding (ECOC) Demo for Classification
This demonstrates how ECOC works for multi-class classification by decomposing
the problem into multiple binary classification tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_wine, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
from itertools import product
import time

# Set random seed for reproducibility
np.random.seed(42)

def load_data(dataset_name='digits'):
    """Load a multi-class dataset for classification"""
    if dataset_name == 'digits':
        # Load digits dataset (10 classes)
        data = load_digits()
        X, y = data.data, data.target
        class_names = [str(i) for i in range(10)]
        print(f"Loaded digits dataset with {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    elif dataset_name == 'wine':
        # Load wine dataset (3 classes)
        data = load_wine()
        X, y = data.data, data.target
        class_names = data.target_names
        print(f"Loaded wine dataset with {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    else:
        # Generate synthetic dataset
        X, y = make_classification(
            n_samples=1000, 
            n_features=20, 
            n_informative=10, 
            n_redundant=5, 
            n_classes=5, 
            random_state=42
        )
        class_names = [f"Class {i}" for i in range(5)]
        print(f"Generated synthetic dataset with {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, class_names

def create_manual_ecoc_matrix(n_classes, coding_type='exhaustive'):
    """Create an ECOC coding matrix manually"""
    if coding_type == 'ovo':
        # One-vs-One coding
        n_classifiers = n_classes * (n_classes - 1) // 2
        coding_matrix = np.zeros((n_classes, n_classifiers))
        
        # Generate all pairs of classes
        pairs = list(product(range(n_classes), repeat=2))
        pairs = [(i, j) for i, j in pairs if i < j]
        
        for classifier_idx, (i, j) in enumerate(pairs):
            coding_matrix[i, classifier_idx] = 1
            coding_matrix[j, classifier_idx] = -1
            
    elif coding_type == 'ovr':
        # One-vs-Rest coding
        coding_matrix = -np.ones((n_classes, n_classes))
        np.fill_diagonal(coding_matrix, 1)
        
    elif coding_type == 'exhaustive' and n_classes <= 7:
        # Exhaustive coding (all possible binary partitions except all 1s and all -1s)
        n_classifiers = 2**(n_classes-1) - 1
        coding_matrix = np.zeros((n_classes, n_classifiers))
        
        # Generate all possible binary strings of length n_classes
        binary_strings = [format(i, f'0{n_classes}b') for i in range(1, 2**n_classes - 1)]
        
        # Filter out strings with all 0s or all 1s
        binary_strings = [s for s in binary_strings if '0' in s and '1' in s]
        
        # Select a subset to avoid redundancy
        selected_strings = binary_strings[:n_classifiers]
        
        for i, binary_string in enumerate(selected_strings):
            for j, bit in enumerate(binary_string):
                coding_matrix[j, i] = 1 if bit == '1' else -1
    
    else:
        # Random coding
        n_classifiers = min(10 * n_classes, 200)  # Limit the number of classifiers
        coding_matrix = np.random.choice([-1, 1], size=(n_classes, n_classifiers))
    
    return coding_matrix

def manual_ecoc_predict(X, binary_classifiers, coding_matrix):
    """Predict using manual ECOC implementation"""
    n_samples = X.shape[0]
    n_classes, n_classifiers = coding_matrix.shape
    
    # Get predictions from all binary classifiers
    predictions = np.zeros((n_samples, n_classifiers))
    for i, clf in enumerate(binary_classifiers):
        predictions[:, i] = clf.predict(X)
    
    # Calculate Hamming distance to each codeword
    distances = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        for j in range(n_classes):
            # Count positions where prediction doesn't match codeword
            distances[i, j] = np.sum(predictions[i, :] != coding_matrix[j, :]) / n_classifiers
    
    # Return class with minimum distance
    return np.argmin(distances, axis=1)

def implement_manual_ecoc(X_train, X_test, y_train, y_test, coding_type='random'):
    """Implement ECOC manually using binary classifiers"""
    n_classes = len(np.unique(y_train))
    
    # Create coding matrix
    coding_matrix = create_manual_ecoc_matrix(n_classes, coding_type)
    n_classifiers = coding_matrix.shape[1]
    
    print(f"\nManual ECOC with {coding_type} coding:")
    print(f"Using {n_classifiers} binary classifiers")
    
    # Train binary classifiers
    binary_classifiers = []
    for i in range(n_classifiers):
        # Create binary labels for this classifier
        binary_y = np.zeros_like(y_train)
        for class_idx in range(n_classes):
            indices = (y_train == class_idx)
            if coding_matrix[class_idx, i] == 1:
                binary_y[indices] = 1
            elif coding_matrix[class_idx, i] == -1:
                binary_y[indices] = 0
            # Ignore classes with 0 in the coding matrix
        
        # Train a binary classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, binary_y)
        binary_classifiers.append(clf)
    
    # Predict using manual ECOC
    y_pred = manual_ecoc_predict(X_test, binary_classifiers, coding_matrix)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Manual ECOC ({coding_type}) Accuracy: {accuracy:.4f}")
    
    return accuracy, coding_matrix

def compare_multiclass_strategies(X_train, X_test, y_train, y_test, class_names):
    """Compare different multiclass classification strategies"""
    results = {}
    confusion_matrices = {}
    
    # Base classifier
    base_clf = LogisticRegression(max_iter=1000, random_state=42)
    
    # One-vs-Rest
    print("\nTraining One-vs-Rest classifier...")
    start_time = time.time()
    ovr = OneVsRestClassifier(base_clf)
    ovr.fit(X_train, y_train)
    ovr_pred = ovr.predict(X_test)
    ovr_time = time.time() - start_time
    ovr_acc = accuracy_score(y_test, ovr_pred)
    results['One-vs-Rest'] = ovr_acc
    confusion_matrices['One-vs-Rest'] = confusion_matrix(y_test, ovr_pred)
    print(f"One-vs-Rest Accuracy: {ovr_acc:.4f} (Time: {ovr_time:.2f}s)")
    
    # One-vs-One
    print("\nTraining One-vs-One classifier...")
    start_time = time.time()
    ovo = OneVsOneClassifier(base_clf)
    ovo.fit(X_train, y_train)
    ovo_pred = ovo.predict(X_test)
    ovo_time = time.time() - start_time
    ovo_acc = accuracy_score(y_test, ovo_pred)
    results['One-vs-One'] = ovo_acc
    confusion_matrices['One-vs-One'] = confusion_matrix(y_test, ovo_pred)
    print(f"One-vs-One Accuracy: {ovo_acc:.4f} (Time: {ovo_time:.2f}s)")
    
    # Error-Correcting Output Codes with different coding strategies
    for coding_strategy in ['random', 'ovr', 'ovo']:
        print(f"\nTraining ECOC classifier with {coding_strategy} coding...")
        start_time = time.time()
        ecoc = OutputCodeClassifier(base_clf, code_size=1.5, random_state=42)
        ecoc.fit(X_train, y_train)
        ecoc_pred = ecoc.predict(X_test)
        ecoc_time = time.time() - start_time
        ecoc_acc = accuracy_score(y_test, ecoc_pred)
        results[f'ECOC ({coding_strategy})'] = ecoc_acc
        confusion_matrices[f'ECOC ({coding_strategy})'] = confusion_matrix(y_test, ecoc_pred)
        print(f"ECOC ({coding_strategy}) Accuracy: {ecoc_acc:.4f} (Time: {ecoc_time:.2f}s)")
    
    # Manual ECOC implementations
    for coding_type in ['ovr', 'ovo', 'random']:
        manual_acc, _ = implement_manual_ecoc(X_train, X_test, y_train, y_test, coding_type)
        results[f'Manual ECOC ({coding_type})'] = manual_acc
    
    # Print detailed report for the best method
    best_method = max(results, key=results.get)
    print(f"\nDetailed report for {best_method}:")
    if 'Manual' in best_method:
        # Re-run the best manual method to get predictions
        coding_type = best_method.split('(')[1].split(')')[0]
        _, coding_matrix = implement_manual_ecoc(X_train, X_test, y_train, y_test, coding_type)
        # Visualize the coding matrix
        plt.figure(figsize=(10, 6))
        sns.heatmap(coding_matrix, cmap='coolwarm', annot=True, fmt='.0f', 
                    xticklabels=[f'C{i+1}' for i in range(coding_matrix.shape[1])],
                    yticklabels=class_names)
        plt.title(f'ECOC Coding Matrix ({coding_type})')
        plt.xlabel('Binary Classifiers')
        plt.ylabel('Classes')
        plt.tight_layout()
        plt.savefig(f'ecoc_coding_matrix_{coding_type}.png')
        plt.close()
        print(f"Coding matrix visualization saved to 'ecoc_coding_matrix_{coding_type}.png'")
    else:
        # Get the predictions from the best sklearn method
        if best_method == 'One-vs-Rest':
            y_pred = ovr_pred
        elif best_method == 'One-vs-One':
            y_pred = ovo_pred
        else:
            y_pred = ecoc_pred
        
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrices[best_method]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {best_method}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('ecoc_confusion_matrix.png')
        plt.close()
        print("Confusion matrix saved to 'ecoc_confusion_matrix.png'")
    
    return results

def compare_base_classifiers(X_train, X_test, y_train, y_test):
    """Compare different base classifiers with ECOC"""
    results = {}
    
    base_classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM (linear)': SVC(kernel='linear', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    }
    
    for name, clf in base_classifiers.items():
        print(f"\nTraining ECOC with {name} as base classifier...")
        ecoc = OutputCodeClassifier(clf, code_size=1.5, random_state=42)
        ecoc.fit(X_train, y_train)
        y_pred = ecoc.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[f'ECOC with {name}'] = acc
        print(f"ECOC with {name} Accuracy: {acc:.4f}")
    
    return results

def visualize_hamming_distances(X_train, y_train, X_test, y_test):
    """Visualize Hamming distances between codewords and predictions"""
    n_classes = len(np.unique(y_train))
    
    # Create coding matrix
    coding_matrix = create_manual_ecoc_matrix(n_classes, 'random')
    n_classifiers = coding_matrix.shape[1]
    
    # Train binary classifiers
    binary_classifiers = []
    for i in range(n_classifiers):
        binary_y = np.zeros_like(y_train)
        for class_idx in range(n_classes):
            indices = (y_train == class_idx)
            if coding_matrix[class_idx, i] == 1:
                binary_y[indices] = 1
            elif coding_matrix[class_idx, i] == -1:
                binary_y[indices] = 0
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, binary_y)
        binary_classifiers.append(clf)
    
    # Get predictions from all binary classifiers
    n_samples = X_test.shape[0]
    predictions = np.zeros((n_samples, n_classifiers))
    for i, clf in enumerate(binary_classifiers):
        predictions[:, i] = clf.predict(X_test)
    
    # Select a few test samples to visualize
    n_samples_to_show = min(5, n_samples)
    sample_indices = np.random.choice(n_samples, n_samples_to_show, replace=False)
    
    # Calculate Hamming distances for selected samples
    plt.figure(figsize=(12, 8))
    
    for idx, sample_idx in enumerate(sample_indices):
        true_class = y_test[sample_idx]
        
        # Calculate Hamming distances to each codeword
        distances = np.zeros(n_classes)
        for class_idx in range(n_classes):
            # Count positions where prediction doesn't match codeword
            distances[class_idx] = np.sum(predictions[sample_idx, :] != coding_matrix[class_idx, :]) / n_classifiers
        
        # Plot distances
        plt.subplot(n_samples_to_show, 1, idx + 1)
        plt.bar(range(n_classes), distances)
        plt.axvline(x=true_class, color='r', linestyle='--', label='True Class')
        plt.xlabel('Class')
        plt.ylabel('Normalized Hamming Distance')
        plt.title(f'Sample {sample_idx}: True Class = {true_class}, Predicted = {np.argmin(distances)}')
        plt.xticks(range(n_classes))
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('ecoc_hamming_distances.png')
    plt.close()
    
    print("\nHamming distance visualization saved to 'ecoc_hamming_distances.png'")

def visualize_error_correction(X_train, y_train, X_test, y_test):
    """Demonstrate error correction capability of ECOC"""
    n_classes = len(np.unique(y_train))
    
    # Create coding matrix with good separation between codewords
    coding_matrix = create_manual_ecoc_matrix(n_classes, 'random')
    n_classifiers = coding_matrix.shape[1]
    
    # Train binary classifiers
    binary_classifiers = []
    for i in range(n_classifiers):
        binary_y = np.zeros_like(y_train)
        for class_idx in range(n_classes):
            indices = (y_train == class_idx)
            if coding_matrix[class_idx, i] == 1:
                binary_y[indices] = 1
            elif coding_matrix[class_idx, i] == -1:
                binary_y[indices] = 0
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, binary_y)
        binary_classifiers.append(clf)
    
    # Get predictions from all binary classifiers for a single test sample
    sample_idx = np.random.randint(0, len(X_test))
    true_class = y_test[sample_idx]
    
    # Get binary predictions
    binary_preds = np.zeros(n_classifiers)
    for i, clf in enumerate(binary_classifiers):
        binary_preds[i] = clf.predict([X_test[sample_idx]])[0]
    
    # Calculate original Hamming distances
    original_distances = np.zeros(n_classes)
    for class_idx in range(n_classes):
        original_distances[class_idx] = np.sum(binary_preds != coding_matrix[class_idx, :]) / n_classifiers
    
    original_prediction = np.argmin(original_distances)
    
    # Introduce errors to binary predictions
    n_errors = min(3, n_classifiers // 4)  # Introduce a few errors
    error_indices = np.random.choice(n_classifiers, n_errors, replace=False)
    corrupted_preds = binary_preds.copy()
    
    for idx in error_indices:
        corrupted_preds[idx] = 1 - corrupted_preds[idx]  # Flip the prediction
    
    # Calculate new Hamming distances
    corrupted_distances = np.zeros(n_classes)
    for class_idx in range(n_classes):
        corrupted_distances[class_idx] = np.sum(corrupted_preds != coding_matrix[class_idx, :]) / n_classifiers
    
    corrupted_prediction = np.argmin(corrupted_distances)
    
    # Visualize the results
    plt.figure(figsize=(15, 10))
    
    # Plot original binary predictions
    plt.subplot(2, 2, 1)
    plt.imshow([binary_preds], cmap='binary', aspect='auto')
    plt.colorbar(ticks=[0, 1])
    plt.title('Original Binary Predictions')
    plt.xlabel('Binary Classifier')
    plt.yticks([])
    
    # Plot corrupted binary predictions
    plt.subplot(2, 2, 2)
    plt.imshow([corrupted_preds], cmap='binary', aspect='auto')
    plt.colorbar(ticks=[0, 1])
    plt.title(f'Corrupted Binary Predictions ({n_errors} errors)')
    plt.xlabel('Binary Classifier')
    plt.yticks([])
    
    # Highlight the errors
    for idx in error_indices:
        plt.plot(idx, 0, 'rx', markersize=10)
    
    # Plot original distances
    plt.subplot(2, 2, 3)
    plt.bar(range(n_classes), original_distances)
    plt.axvline(x=true_class, color='g', linestyle='--', label='True Class')
    plt.axvline(x=original_prediction, color='r', linestyle='-', label='Predicted Class')
    plt.xlabel('Class')
    plt.ylabel('Normalized Hamming Distance')
    plt.title(f'Original Distances: Predicted = {original_prediction}')
    plt.xticks(range(n_classes))
    plt.legend()
    
    # Plot corrupted distances
    plt.subplot(2, 2, 4)
    plt.bar(range(n_classes), corrupted_distances)
    plt.axvline(x=true_class, color='g', linestyle='--', label='True Class')
    plt.axvline(x=corrupted_prediction, color='r', linestyle='-', label='Predicted Class')
    plt.xlabel('Class')
    plt.ylabel('Normalized Hamming Distance')
    plt.title(f'Corrupted Distances: Predicted = {corrupted_prediction}')
    plt.xticks(range(n_classes))
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ecoc_error_correction.png')
    plt.close()
    
    print("\nError correction visualization saved to 'ecoc_error_correction.png'")
    if corrupted_prediction == true_class and original_prediction == true_class:
        print("ECOC successfully maintained correct prediction despite binary classifier errors!")
    elif corrupted_prediction == true_class and original_prediction != true_class:
        print("ECOC corrected the prediction after introducing errors (unusual but possible)!")
    elif corrupted_prediction != true_class and original_prediction == true_class:
        print("ECOC failed to maintain correct prediction after introducing errors.")
    else:
        print("Both original and corrupted predictions were incorrect.")

def study_code_size_effect(X_train, X_test, y_train, y_test):
    """Study the effect of code size on ECOC performance"""
    code_sizes = [0.5, 1.0, 1.5, 2.0, 3.0]
    accuracies = []
    
    for code_size in code_sizes:
        ecoc = OutputCodeClassifier(
            LogisticRegression(max_iter=1000, random_state=42),
            code_size=code_size,
            random_state=42
        )
        ecoc.fit(X_train, y_train)
        y_pred = ecoc.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(code_sizes, accuracies, marker='o')
    plt.xlabel('Code Size')
    plt.ylabel('Accuracy')
    plt.title('Effect of Code Size on ECOC Performance')
    plt.grid(True)
    plt.savefig('ecoc_code_size.png')
    plt.close()
    
    print("\nCode size effect study saved to 'ecoc_code_size.png'")

if __name__ == "__main__":
    print("Error-Correcting Output Coding (ECOC) Demo for Classification")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test, class_names = load_data(dataset_name='digits')
    
    # Compare multiclass strategies
    results = compare_multiclass_strategies(X_train, X_test, y_train, y_test, class_names)
    
    # Compare base classifiers
    base_clf_results = compare_base_classifiers(X_train, X_test, y_train, y_test)
    results.update(base_clf_results)
    
    # Visualize Hamming distances
    visualize_hamming_distances(X_train, y_train, X_test, y_test)
    
    # Demonstrate error correction
    visualize_error_correction(X_train, y_train, X_test, y_test)
    
    # Study code size effect
    study_code_size_effect(X_train, X_test, y_train, y_test)
    
    # Visualize results comparison
    methods = list(results.keys())
    accuracies = list(results.values())
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(methods, accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(methods))))
    plt.ylabel('Accuracy')
    plt.title('Comparison of Multi-class Classification Methods')
    plt.ylim(min(accuracies) - 0.05, 1.0)
    plt.xticks(rotation=45, ha='right')
    
    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center', rotation=90)
    
    plt.tight_layout()
    plt.savefig('ecoc_comparison.png')
    plt.close()
    
    print("\nAccuracy comparison saved to 'ecoc_comparison.png'")
    print("\nECOC demo completed!")
    
    # Additional explanation
    print("\nKey Insights about Error-Correcting Output Coding (ECOC):")
    print("1. ECOC decomposes multi-class problems into multiple binary classification tasks.")
    print("2. Each class is assigned a unique binary code (codeword).")
    print("3. The coding matrix defines which classes participate in each binary task.")
    print("4. Prediction uses Hamming distance to find the closest codeword.")
    print("5. ECOC can correct errors from individual binary classifiers.")
    print("6. Longer codes generally provide better error correction but require more computation.")
    print("7. Common coding strategies include one-vs-rest, one-vs-one, and random coding.")
    print("8. ECOC is particularly useful for problems with many classes.")
