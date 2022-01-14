from torch_geometric import data, utils
from torch_geometric.utils import to_networkx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.transforms import LineGraph, Delaunay 
from torch_geometric.data import Data
import torch
import networkx as nx
import itertools


def graph_show_plt(data, res_name):
    pts = data['pos']
    face = data['face']
    plt.plot(pts[:,0], pts[:,1], 'o')
    plt.triplot(pts[:,0], pts[:, 1], face.t())
    plt.savefig(f'./images/{res_name}.png')


def graph_show_nx(data, res_name, custom_pos=True):
    '''
    show torchgemotric graph by networkx
    args:
        custom_pos: use node pos to customize graph layout
    '''
    plt.clf()
    G = to_networkx(data, to_undirected=True)
    labels = {(u, v): chr(ord('A')+ind)  for ind, (u, v, d) in enumerate(G.edges(data=True))}
    pts = data['pos']
    pos=nx.shell_layout(G)
    if custom_pos:
        for ind, (key, vlaue) in enumerate(pos.items()):
            pos[key] = pts[ind].numpy()
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    nx.draw(G, pos, with_labels=True, node_color='#CCCCCC', font_color='blue')
    plt.savefig(f'./images/{res_name}.png')


def linegraph_show_nx(data, res_name):
    '''
    show torchgemotric graph by networkx
    args:
        custom_pos: use node pos to customize graph layout
    '''
    plt.clf()
    G = to_networkx(data, to_undirected=True)
    node_labels = {n: chr(ord('A')+n) for n in G.nodes()}
    pos=nx.shell_layout(G)
    nx.draw(G, pos, labels=node_labels, node_color='#CCCCCC', font_color='blue')
    
    plt.savefig(f'./images/{res_name}.png')

def simplies2edges(data: Data):
    '''
    convert simplices to edge_index: 2 * n
    '''
    n = data['pos'].size(0)
    A = np.zeros((n, n))
    for simplex in data['face'].t():
        for pair in itertools.permutations(simplex, 2):
            A[pair] = 1
    data['edge_index'] = torch.LongTensor(np.stack(np.where(A>0)))
    return data

    
if __name__ == "__main__":

    # points = np.random.rand(10).reshape(5, 2)*256
    num_nodes = 10
    x = torch.rand(num_nodes*8).reshape(num_nodes,8)
    # points = np.array([[1,1],[3,1], [0,0], [2,0.2], [4,0]])
    points = np.random.random((num_nodes,2)) * 10
    
    data = Data(x=x, edge_index=None, pos=torch.FloatTensor(points), num_nodes=num_nodes)
    
    # generate delaunay graph
    delaunay_trans = Delaunay()
    data = delaunay_trans(data)
    data = simplies2edges(data)
    graph_show_nx(data, 'graph')

    
    # generate line graph
    lg = LineGraph()
    line_graph_data = lg(data)
    linegraph_show_nx(line_graph_data, 'linegraph')  