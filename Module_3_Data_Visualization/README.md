
# Lecture 3: Data Visualization and Multi-Platform Analysis

## Overview
This lecture demonstrates various approaches to data visualization and analysis across different platforms, including Python data generation, statistical analysis, and graph database visualization using Neo4j.

## Data Generation and Excel Export

### Creating Sample Data
In xlsx_data_generation.py, we generate synthetic production data using mixed normal distributions:
```python
distribution1 = np.random.normal(loc=50, scale=5, size=15)  # N(50,5)
distribution2 = np.random.normal(loc=100, scale=20, size=85)  # N(100,20)
```
This creates a realistic production dataset with two distinct patterns.

### Data Export
The program exports data to both Excel and creates initial visualizations:
- Saves production data to Excel format
- Generates a basic histogram visualization
- Maintains data integrity for further analysis

## Advanced Python Visualization

### Statistical Analysis
In python_visualization.py, we perform more sophisticated analysis:
- Histogram generation with enhanced styling
- Gaussian Mixture Model (GMM) fitting
- Probability density visualization
- CSV conversion for database import

### Visualization Techniques
The code demonstrates multiple visualization approaches:
- Basic histograms with customized parameters
- GMM component plotting
- Density estimation curves
- Statistical summary generation

## Neo4j Graph Visualization

### Data Import and Relationships
In neo4j_visualization.cypher, we create a graph structure:
- Factory and Product nodes
- Supplier relationships
- Production record relationships
- Distribution network modeling

### Important Note
The Cypher code requires the CSV file to be in the correct directory for Neo4j import. You may need to:
1. Move the generated CSV file to Neo4j's import directory
2. Update the file path in the LOAD CSV command
3. Ensure proper permissions are set

## Conclusion
This multi-platform approach demonstrates how different tools and visualization techniques can provide comprehensive insights into production data, from statistical analysis to graph-based relationship modeling.
