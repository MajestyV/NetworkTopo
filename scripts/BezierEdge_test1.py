# __author__ = 'jmh081701'
import matplotlib.pyplot as plt
import numpy as np
import bezier
from matplotlib.collections import LineCollection
import networkx as nx
import random

G = nx.watts_strogatz_graph(40, 4, 0.2)

# G = nx.Graph()
# G.add_nodes_from([0,1,2,3])
# G.add_edge(0,1,weight=0.1)

# G.add_edge(1,2,weight=0.2)

# G.add_edge(0,2,weight=0.3)
# G.add_edge(1,3,weight=0.4)
graph = G
def curved_line(x0, y0, x1, y1, eps=0.2, pointn=30):

    x2 = (x0+x1)/2.0 + 0.1 ** (eps+abs(x0-x1)) * (-1) ** (random.randint(1,4))
    y2 = (y0+y1)/2.0 + 0.1 ** (eps+abs(y0-y1)) * (-1) ** (random.randint(1,4))
    nodes = np.asfortranarray([
        [x0, x2, x1],
        [y0, y2, y1]
    ])
    curve = bezier.Curve(nodes,
                         degree=2)
    s_vals = np.linspace(0.0, 1.0, pointn)
    data=curve.evaluate_multi(s_vals)
    x=data[0]
    y=data[1]
    segments =[]
    for index in range(0,len(x)):
        segments.append([x[index],y[index]])
    segments = [segments]
    return  segments
def curved_graph(_graph, pos = None, eps=0.2, pointn=30):

    if pos == None:
        pos = nx.spring_layout(graph)

    for u,v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        segs = curved_line(x0,y0,x1,y1)
        lc = LineCollection(segs)
        plt.gca().add_collection(lc)
        plt.gca().autoscale_view()

if __name__ == '__main__':
    #画节点
    # pos = nx.spring_layout(graph)
    pos = nx.circular_layout(G)  # Position nodes on a circle

    # nx.draw_networkx_nodes(G,pos, with_label=True)
    # nx.draw_networkx_edges(G,pos)
    # plt.show()
    #画曲线
    nx.draw_networkx_nodes(G,pos)
    curved_graph(graph,pos)

    plt.show()


