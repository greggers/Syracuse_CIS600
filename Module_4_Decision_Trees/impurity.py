import math
from collections import Counter

def gini_impurity(labels):
    total = len(labels)
    counts = Counter(labels)
    return 1 - sum((count / total) ** 2 for count in counts.values())

def entropy_impurity(labels):
    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values() if count != 0)