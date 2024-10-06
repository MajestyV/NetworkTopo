# Imports
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
# from fa2 import ForceAtlas2
from curved_edges import curved_edges

# Load the graph edges and compute the node positions using ForceAtlas2
# G = nx.read_edgelist('facebook_combined.txt')
# forceatlas2 = ForceAtlas2()
# positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=50)
G = nx.watts_strogatz_graph(40, 4, 0.2)
positions = nx.circular_layout(G)  # Position nodes on a circle

# Produce the curves
curves = curved_edges(G, positions)
lc = LineCollection(curves, color='w', alpha=0.05)

# Plot
plt.figure(figsize=(20,20))
nx.draw_networkx_nodes(G, positions, node_size=5, node_color='w', alpha=0.4)
plt.gca().add_collection(lc)
plt.tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
plt.show()