import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def R_shuffle(node_number=0,path_length=0):
    x = [np.random.random() for i in range(path_length)]
    e = [int(i / sum(x) * (node_number-path_length)) + 1 for i in x]
    re = node_number - sum(e)
    u = [np.random.randint(0, path_length - 1) for i in range(re)]
    for i in range(re):
        e[u[i]] += 1
    return e

def Network_initial(network_name=None, network_size=300, density=0.2, Depth=10, MC_configure=None, random_seed=2048):
    rng = np.random.RandomState(random_seed)
    if network_name == "ER":
        rg = nx.erdos_renyi_graph(network_size, density, directed=False)  # ER
        R_initial = nx.adjacency_matrix(rg).toarray()
    elif network_name == "DCG":
        rg = nx.erdos_renyi_graph(network_size, density, directed=True)  # ER
        R_initial = nx.adjacency_matrix(rg).toarray()
    elif network_name == "DAG":
        if MC_configure is not None:
            xx = np.append(0, np.cumsum(MC_configure['number']))
            for i in range(xx.shape[0] - 1):
                Reject_index = 1
                for j in range(0, xx.shape[0] - 1):
                    if len(MC_configure[i + 1]) == np.sum(np.isin(MC_configure[i + 1], MC_configure[j + 1] + 1)):
                        Reject_index = 0
                if Reject_index == 1 and (MC_configure[i + 1] != 1).all():
                    print("fail to construct the DAN under current Memory commnity strcutrue configuration")
                    Reject_index = 2
            if Reject_index != 2:
                R_initial_0 = np.zeros((network_size, network_size))
                for i in range(xx.shape[0] - 1):
                    for j in range(xx.shape[0] - 1):
                        if len(MC_configure[i + 1]) == np.sum(np.isin(MC_configure[i + 1] + 1, MC_configure[j + 1])):
                            R_initial_0[xx[i]:xx[i + 1], xx[j]:xx[j + 1]] = 1
                R_initial = np.triu(R_initial_0, 1)
            else:
                R_initial = None

        else:
            xx = R_shuffle(network_size, Depth)
            # xx=np.array([3,4,3])
            # xx=np.array([60,60,60,60,60])
            # xx=np.array([30,30,30,30,30,30,30,30,30,30])*3
            rg = nx.complete_multipartite_graph(*tuple(xx))
            x = nx.adjacency_matrix(rg).toarray()
            R_initial = np.triu(x, 1)
            # R_initial= np.tril(x,1)

        Real_density = np.sum(R_initial > 0) * 1.0 / (network_size ** 2)
        if Real_density > 0 and density < Real_density:
            R_initial[rng.rand(*R_initial.shape) <= (1.0 - density / Real_density)] = 0
        R_initial = np.triu(R_initial, 1)
    return R_initial

def lattice_layout(L: int) -> dict:
    pos = {}
    for i in np.arange(L * L):
        pos[i] = (i / L, i % L)
    return pos

class PlotNetworkTopo:
    '''
    Plot the network topology
    '''

    def __init__(self, network_topo: str, network_dim: int, network_den: float, k: int, random_seed: int=None, **kwargs):
        '''
        :param network_topo:
        :param network_dim: dimension of the network
        :param network_den: density of the network
        :param k: number of connected nodes
        '''
        self.network_topo = network_topo

        if random_seed is not None:
            np.random.RandomState(random_seed)  # 根据种子产生随机数列，保证每个迭代都产生固定随机序列
        else:
            pass

        if network_topo == 'ER':
            rg = nx.erdos_renyi_graph(network_dim, network_den, directed=False)  # ER
        elif network_topo == 'lattice_2D':
            if kwargs['L']*kwargs['W'] != network_dim:
                raise ValueError('The dimension of the lattice network is not correct!')
            else:
                rg = nx.grid_2d_graph(kwargs['L'], kwargs['W'])  # Grid 2D
        elif network_topo == 'ring':
            rg = nx.cycle_graph(network_dim)
        elif network_topo == 'BA':
            rg = nx.barabasi_albert_graph(network_dim, k)
        elif network_topo == 'WS':
            rg = nx.watts_strogatz_graph(network_dim, k, network_den)
        else:
            raise ValueError('The network topology is not supported!')

        self.random_graph = rg

    def Show(self):
        '''
        Show the network topology (可视化函数)
        '''
        # 可视化
        # Seed for reproducible layout (# 获取节点位置)
        # pos = dict((n, n) for n in rg.nodes())  # lattice grid (2D)
        if self.network_topo == 'ER':
            pos = nx.spring_layout(self.random_graph, seed=random_seed)  # Position nodes using Fruchterman-Reingold force-directed algorithm (Works for grid_2D as well)

        elif self.network_topo == 'lattice_2D':
            pos = dict((n, n) for n in self.random_graph.nodes())  # lattice grid (2D)

        elif (self.network_topo == 'ring') or (self.network_topo == 'BA') or (self.network_topo == 'WS'):
            pos = nx.circular_layout(self.random_graph)  # Position nodes on a circle

        else:
            raise ValueError('The network topology is not supported!')

        nx.draw(self.random_graph, pos=pos)

        return

        # or (self.network_topo == 'lattice_2D') or (self.network_topo == 'BA')

        # pos = nx.spring_layout(rg, seed=random_seed)  # Position nodes using Fruchterman-Reingold force-directed algorithm (Works for grid_2D as well)
        # pos = nx.circular_layout(rg)  # Position nodes on a circle
        # pos = nx.shell_layout(rg)  # Position nodes in concentric circles
        # pos = nx.spectral_layout(rg)  # Position nodes using the eigenvectors of the graph Laplacian
        # pos = nx.planar_layout(rg)  # Position nodes without edge crossings


if __name__ == '__main__':
    random_seed = 2048
    reservoir_dimension = 36
    reservoir_density = 0.2
    network_mode = 'BA'

    network_gragh = PlotNetworkTopo(network_topo=network_mode, network_dim=reservoir_dimension, network_den=reservoir_density,
                                    k=4, random_seed=random_seed, L=6, W=6)

    network_gragh.Show()




    # pos = lattice_layout(int(np.sqrt(reservoir_dimension)))


    # nx.draw(rg, pos=pos)
    # nx.draw(rg,pos=pos)
    plt.show()