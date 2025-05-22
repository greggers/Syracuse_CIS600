# Data Science Course Materials

This repository contains lecture materials, code examples, and visualizations for a comprehensive data science course.

## Course Structure

### Lecture 1: Data Science Overview
An introduction to the field of data science, covering fundamental concepts, methodologies, and applications. This lecture provides a broad overview of what data science is and how it's applied across various domains.

### Lecture 2: Data Sampling Methods
A deep dive into data types, structures, and sources. This lecture explores how data is collected, organized, and prepared for analysis, with a focus on understanding data characteristics and quality.

### Lecture 3: Data Visualization
A comprehensive guide to data visualization techniques, including using Excel, matplotlib, and graph databases. This lecture covers the art of presenting data in a clear, concise, and visually appealing manner.

### Lecture 4: Decision Trees
A detailed exploration of decision trees, their construction using ID3 and CART algorithms, and their applications in classification and regression tasks. This lecture delves into the principles and practical applications of decision trees.

### Lecture 5: Model Evaluation and Hyperparameter Selection
A comprehensive guide to evaluating and selecting models, including techniques for hyperparameter tuning and model selection. This lecture covers the importance of model evaluation and how to choose the right model for a given problem. With selections discussing, 
resubstitution, cross-validation, and other methods.

### Lecture 6: Ensemble Methods - Bagging and Boosting
An in-depth exploration of ensemble learning techniques that combine multiple models to improve prediction accuracy and robustness. This lecture covers:

- **Bagging (Bootstrap Aggregating)**: Creating multiple versions of a predictor by training on random subsets of the data with replacement, then aggregating their predictions.
  - Random Forests: An extension of bagging that uses decision trees as base learners with random feature selection.
  - Variance reduction and stability improvements through model averaging.

- **Boosting**: Sequential ensemble methods that iteratively train models to correct errors made by previous models.
  - AdaBoost: Adaptive Boosting that adjusts sample weights based on previous model errors.
  - Gradient Boosting: Building models that optimize a differentiable loss function.
  - XGBoost, LightGBM, and CatBoost: Modern implementations with performance optimizations.

- **Comparison of Ensemble Methods**: Analysis of when to use bagging vs. boosting, with considerations for bias-variance tradeoff, computational complexity, and hyperparameter tuning.

### Lecture 7: Association Rules and Market Basket Analysis
A comprehensive introduction to association rule mining and its applications in discovering relationships between variables in large datasets. This lecture covers:

- **Association Rule Fundamentals**: Understanding the "if-then" statements that help show probability relationships between items in transactional databases.
  - Rule format: {X} → {Y} (If you see set X, how often do you see set Y)
  - Applications in retail, web analytics, and recommendation systems.

- **Key Metrics for Association Rules**:
  - Support: How often items appear together in the data - s(X→Y) = (σ(X ∪ Y))/(N)
  - Confidence: How often the rules are true - c(X → Y) = (σ(X ∪ Y))/(σ(X))
  - Lift: Ratio of confidence vs. inferred response - l(X → Y) = c(X→Y)/s(Y)
  - Additional metrics: Leverage, Conviction, and Zhang's metric for balanced assessment

## Code Examples

The repository includes Python code examples demonstrating key data science concepts:

- **Curse of Dimensionality**: Visualizations showing how the distribution of points changes as dimensions increase, using Gaussian distributions in 1D and 2D spaces.
- **Sampling Methods**: Implementations of uniform random sampling and Monte Carlo sampling with visualizations to demonstrate their effectiveness in different scenarios.
- **Figure Generation**: An example of generating figures using matplotlib and scikit, showcasing how to create informative and visually appealing plots.
- **Decision Tree Example**: A simple implementation of a decision tree for classification tasks, demonstrating how to build and use decision trees for predictive modeling.
- **Ensemble Methods**: Implementation of bagging and boosting algorithms, including Random Forests and Gradient Boosting, with comparisons of their performance on various datasets.
- **Association Rule Mining**: Examples of market basket analysis using the Apriori algorithm, with calculations of support, confidence, lift, and other metrics to evaluate rule strength.

## Getting Started

### Prerequisites
- Python 3.x
- Required packages: numpy, matplotlib, scipy

### Installation
```bash
pip install numpy matplotlib scipy
```

### Running the Examples
Each Python script can be executed directly:
```bash
python curse_of_dimensionality.py
python sampling_visualization.py
```

## Visualizations

The code examples generate visualizations to help understand complex data science concepts:

- Distribution of points in different dimensional spaces
- Comparison of sampling techniques
- Function estimation using different sampling methods
- Decision tree construction and visualization
- Performance comparison of ensemble methods
- Association rule metrics and relationships

## Contributing

Feel free to submit pull requests or open issues to improve the course materials or add new examples.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
