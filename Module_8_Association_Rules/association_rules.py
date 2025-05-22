import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def generate_grocery_transactions(num_transactions=1000):
    """
    Generate synthetic grocery store transaction data
    """
    # Define product categories and items
    products = {
        'dairy': ['milk', 'cheese', 'yogurt', 'butter'],
        'bakery': ['bread', 'bagels', 'muffins', 'cookies'],
        'produce': ['apples', 'bananas', 'lettuce', 'tomatoes', 'carrots'],
        'meat': ['chicken', 'beef', 'pork', 'fish'],
        'beverages': ['soda', 'juice', 'water', 'coffee', 'tea'],
        'snacks': ['chips', 'pretzels', 'nuts', 'chocolate', 'candy']
    }
    
    # Flatten the list of products
    all_products = [item for category in products.values() for item in category]
    
    # Generate transactions
    transactions = []
    for _ in range(num_transactions):
        # Determine number of items in this transaction (between 1 and 10)
        num_items = np.random.randint(1, 11)
        
        # Create some common associations
        if np.random.random() < 0.7 and num_items >= 2:
            # 70% chance of bread and milk appearing together
            transaction = ['bread', 'milk']
            num_items -= 2
        elif np.random.random() < 0.6 and num_items >= 2:
            # 60% chance of chips and soda appearing together
            transaction = ['chips', 'soda']
            num_items -= 2
        elif np.random.random() < 0.5 and num_items >= 3:
            # 50% chance of chicken, lettuce, and bread appearing together
            transaction = ['chicken', 'lettuce', 'bread']
            num_items -= 3
        else:
            transaction = []
        
        # Add random products to reach the desired transaction size
        remaining_products = [p for p in all_products if p not in transaction]
        transaction.extend(np.random.choice(remaining_products, 
                                           size=min(num_items, len(remaining_products)), 
                                           replace=False))
        
        transactions.append(transaction)
    
    return transactions

def create_one_hot_encoded_df(transactions, all_items):
    """
    Convert transactions to one-hot encoded DataFrame
    """
    # Create a DataFrame with one-hot encoded values
    one_hot_encoded = pd.DataFrame([[item in transaction for item in all_items] 
                                   for transaction in transactions], 
                                   columns=all_items)
    return one_hot_encoded

def calculate_manual_metrics(transactions, X, Y):
    """
    Manually calculate support, confidence, and lift for rule X -> Y
    
    Parameters:
    - transactions: list of transactions
    - X: antecedent items (list)
    - Y: consequent items (list)
    
    Returns:
    - support, confidence, lift
    """
    N = len(transactions)
    
    # Count transactions containing X
    X_count = sum(1 for transaction in transactions if all(item in transaction for item in X))
    
    # Count transactions containing Y
    Y_count = sum(1 for transaction in transactions if all(item in transaction for item in Y))
    
    # Count transactions containing both X and Y
    XY_count = sum(1 for transaction in transactions if all(item in transaction for item in X) and 
                                                       all(item in transaction for item in Y))
    
    # Calculate metrics
    support = XY_count / N
    confidence = XY_count / X_count if X_count > 0 else 0
    lift = confidence / (Y_count / N) if Y_count > 0 else 0
    
    return support, confidence, lift

def main():
    # Generate synthetic grocery transactions
    print("Generating synthetic grocery transaction data...")
    transactions = generate_grocery_transactions(1000)
    
    # Display a few sample transactions
    print("\nSample transactions:")
    for i in range(5):
        print(f"Transaction {i+1}: {transactions[i]}")
    
    # Get all unique items
    all_items = sorted(set(item for transaction in transactions for item in transaction))
    
    # Create one-hot encoded DataFrame for apriori algorithm
    one_hot_df = create_one_hot_encoded_df(transactions, all_items)
    
    # Find frequent itemsets using Apriori algorithm
    print("\nFinding frequent itemsets using Apriori algorithm...")
    frequent_itemsets = apriori(one_hot_df, min_support=0.05, use_colnames=True)
    
    # Generate association rules
    print("Generating association rules...")
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    # Sort rules by lift
    rules = rules.sort_values('lift', ascending=False)
    
    # Display top 10 rules
    print("\nTop 10 association rules by lift:")
    pd.set_option('display.max_columns', None)
    print(rules.head(10))
    
    # Manual calculation example
    print("\nManual calculation example:")
    X = ['bread']
    Y = ['milk']
    manual_support, manual_confidence, manual_lift = calculate_manual_metrics(transactions, X, Y)
    
    print(f"\nRule: {X} -> {Y}")
    print(f"Support: {manual_support:.4f}")
    print(f"Confidence: {manual_confidence:.4f}")
    print(f"Lift: {manual_lift:.4f}")
    
    # Explanation of metrics
    print("\nExplanation of metrics:")
    print(f"Support({X} -> {Y}) = count({X} ∪ {Y}) / N")
    print(f"  = The probability of finding both {X} and {Y} together in transactions")
    
    print(f"\nConfidence({X} -> {Y}) = count({X} ∪ {Y}) / count({X})")
    print(f"  = The probability of finding {Y} when {X} is present")
    
    print(f"\nLift({X} -> {Y}) = Confidence({X} -> {Y}) / Support({Y})")
    print(f"  = How much more likely {Y} is purchased when {X} is purchased")
    print(f"  = A lift > 1 indicates {X} and {Y} appear together more often than expected by chance")
    
    # Visualize top rules
    plt.figure(figsize=(10, 6))
    top_rules = rules.head(10)
    sns.barplot(x=top_rules.index, y='lift', data=top_rules)
    plt.title('Top 10 Association Rules by Lift')
    plt.xlabel('Rule Index')
    plt.ylabel('Lift Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('top_rules_lift.png')
    
    # Scatter plot of support vs confidence
    plt.figure(figsize=(10, 6))
    plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Support vs Confidence for Association Rules')
    plt.grid(True)
    plt.savefig('support_vs_confidence.png')
    
    print("\nVisualization files 'top_rules_lift.png' and 'support_vs_confidence.png' have been saved.")

if __name__ == "__main__":
    main()