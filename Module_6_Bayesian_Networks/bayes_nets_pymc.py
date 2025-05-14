from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import matplotlib.pyplot as plt

# Define the Bayesian network structure
model = BayesianModel([('Difficulty', 'Grade'), ('Intelligence', 'Grade'),
                        ('Grade', 'Letter'), ('Grade', 'SAT')])

# Define the conditional probability distributions (CPDs)
cpd_d = TabularCPD(variable='Difficulty', variable_card=2, 
                    values=[[0.6], [0.4]],
                    state_names={'Difficulty': ['easy', 'hard']})

cpd_i = TabularCPD(variable='Intelligence', variable_card=2, 
                    values=[[0.7], [0.3]],
                    state_names={'Intelligence': ['low', 'high']})

cpd_g = TabularCPD(variable='Grade', variable_card=3,
                    values=[[0.3, 0.6, 0.1, 0.3],
                            [0.4, 0.3, 0.6, 0.4],
                            [0.3, 0.1, 0.3, 0.3]],
                    evidence=['Difficulty', 'Intelligence'],
                    evidence_card=[2, 2],
                    state_names={'Grade': ['A', 'B', 'C'],
                                 'Difficulty': ['easy', 'hard'],
                                 'Intelligence': ['low', 'high']})

cpd_l = TabularCPD(variable='Letter', variable_card=2,
                    values=[[0.1, 0.4, 0.9],
                            [0.9, 0.6, 0.1]],
                    evidence=['Grade'],
                    evidence_card=[3],
                    state_names={'Letter': ['strong', 'weak'],
                                 'Grade': ['A', 'B', 'C']})

cpd_s = TabularCPD(variable='SAT', variable_card=2,
                    values=[[0.95, 0.2, 0.05],
                            [0.05, 0.8, 0.95]],
                    evidence=['Grade'],
                    evidence_card=[3],
                    state_names={'SAT': ['high', 'low'],
                                 'Grade': ['A', 'B', 'C']})

# Add CPDs to the model
model.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)

# Check model consistency
model.check_model()

# Visualize the network
pos = nx.circular_layout(model)
nx.draw(model, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold")
plt.show()