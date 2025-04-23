# Module 4: Decision Trees

This module introduces the concept of Decision Trees using the classic "Play Tennis" dataset. The content is designed for graduate-level computer science students and progresses through theoretical foundations, implementations, and visual illustrations.

## Contents

### Step 1: Gini and Entropy
- Implements Gini and Entropy impurity functions.
- Visualizes impurity metrics using the "Play Tennis" dataset.
- Files: `impurity.py`, `impurity_plot.py`

### Step 2: CART Algorithm
- Builds and visualizes a CART decision tree.
- Uses matplotlib to highlight decision splits.
- File: `cart_tree.py`

### Step 3: ID3 Algorithm
- Builds and visualizes an ID3 tree using information gain.
- File: `id3_tree.py`

### Step 4: Tree Pruning
- Demonstrates pruning on the CART tree.
- Includes a plot showing accuracy vs. tree complexity.
- File: `pruning.py`

## Dataset

The dataset used in this module is a simplified version of the UCI "Play Tennis" dataset. Each row represents a day, with weather conditions and a label indicating whether tennis was played:

| Outlook  | Temperature | Humidity | Wind   | PlayTennis |
|----------|-------------|----------|--------|------------|
| Sunny    | Hot         | High     | Weak   | No         |
| Sunny    | Hot         | High     | Strong | No         |
| Overcast | Hot         | High     | Weak   | Yes        |
| Rain     | Mild        | High     | Weak   | Yes        |
| Rain     | Cool        | Normal   | Weak   | Yes        |
| Rain     | Cool        | Normal   | Strong | No         |
| Overcast | Cool        | Normal   | Strong | Yes        |
| Sunny    | Mild        | High     | Weak   | No         |
| Sunny    | Cool        | Normal   | Weak   | Yes        |
| Rain     | Mild        | Normal   | Weak   | Yes        |
| Sunny    | Mild        | Normal   | Strong | Yes        |
| Overcast | Mild        | High     | Strong | Yes        |
| Overcast | Hot         | Normal   | Weak   | Yes        |
| Rain     | Mild        | High     | Strong | No         |

This dataset is commonly used to illustrate how decision trees make decisions based on discrete feature values and is generated in the `play_tennis_data.py` file.

## Requirements

```bash
pip install matplotlib
```
