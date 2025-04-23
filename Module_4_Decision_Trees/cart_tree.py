import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from play_tennis_data import load_play_tennis

# Load dataset
data = load_play_tennis()
df = pd.DataFrame(data)

# Encode categorical features
le_dict = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    le_dict[column] = le

# Prepare features and labels
X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

# Train CART decision tree
clf = DecisionTreeClassifier(criterion='gini', max_depth=None)
clf.fit(X, y)

# Plot tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=le_dict['PlayTennis'].classes_, filled=True)
plt.title("CART Decision Tree for Play Tennis Dataset")
plt.savefig("cart_tree_plot.png")
plt.show()
