import matplotlib.pyplot as plt
from impurity import gini_impurity, entropy_impurity
from play_tennis_data import load_play_tennis

def plot_impurities():
    data = load_play_tennis()
    class_labels = [item['PlayTennis'] for item in data]

    gini = gini_impurity(class_labels)
    entropy = entropy_impurity(class_labels)

    plt.figure(figsize=(6, 4))
    plt.bar(['Gini', 'Entropy'], [gini, entropy], color=['skyblue', 'orange'])
    plt.title("Gini vs Entropy Impurity for 'Play Tennis'")
    plt.ylabel("Impurity")
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("impurity_plot.png")
    plt.show()

if __name__ == "__main__":
    plot_impurities()
