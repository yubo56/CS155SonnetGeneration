import pickle
import networkx as nx
import numpy as np
from HMM import *
import matplotlib.pyplot as plt

# Load HMM object
#with open('full.pkl', 'rb') as f:
with open('full10.pkl', 'rb') as f:
    h = pickle.load(f)

# Matrix and size
matrix = h.A
size = matrix[0].size

# Create a graph
DG = nx.DiGraph()
DG.add_nodes_from(range(size))

# Add edges and their weights
for i in range(size):
    for j in range(size):
        weight = matrix.item((i, j))
        if round(weight, 3) != 0.0:
            DG.add_edge(i, j, weight=matrix.item((i, j)))

# Draw the graph
pos = nx.circular_layout(DG)
nx.draw(DG, pos, with_labels=True)
edge_labels=dict([((u, v, ), '%.2f'%(d['weight'])) for u, v, d in DG.edges(data=True)])
nx.draw_networkx_edge_labels(DG, pos, edge_labels=edge_labels, label_pos=0.27, font_size=7)
plt.axis('off')
plt.show()