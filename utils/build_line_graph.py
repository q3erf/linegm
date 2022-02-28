from itertools import zip_longest
from tkinter.messagebox import NO
import torch
from torch_geometric.data import Data
import numpy as np
from torch_sparse import coalesce


def build_line_graph(data: Data, perm):
    N = data.num_nodes
    # print('graph nodes num: ', N)
    edge_index, edge_attr = data.edge_index, data.edge_attr
    x = data.x
    edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
    
    mask = []
    edge_index = edge_index.T
    for edge in edge_index:
        if edge[0] > edge[1]:
            mask.append(False)
        else:
            mask.append(True)
    edge_index = edge_index[mask]
    line_num_nodes = edge_index.size(0)
    # print('line graph nodes num: ', line_num_nodes)
    # print(edge_index)

    line_edge_index = []
    line_x = []

    for (r,c) in edge_index:
        line_x.append(np.concatenate((x[r.item()], x[c.item()]), axis=0))

    for node1, (r1,c1) in enumerate(edge_index):
        for ind, (r2,c2) in enumerate(edge_index[node1+1:]):
            # 若共享一个节点 那么连接边
            node2 = ind+node1+1
            #     line_perm[(node1, node2)] = 1
            if r1==r2 or r1==c2 or c1==r2 or c1==c2:
                line_edge_index.append([node1, node2])

    # print('line_edge_index: ', line_edge_index)
    
    linegraph = Data(
        x = torch.FloatTensor(line_x),
        edge_index = torch.LongTensor(line_edge_index).t(),
        x_token = edge_index.T,
    )
    linegraph.num_nodes = line_num_nodes
    # print(linegraph)

    return linegraph

def build_line_perm(edge_index_g1, edge_index_g2, perm):
    line_correspond = []
    edge_index_g1 = edge_index_g1.T
    edge_index_g2 = edge_index_g2.T

    N1, N2 = edge_index_g1.size(0), edge_index_g2.size(0)
    # print('size: ', N1, N2)
    N = max(N1, N2)
    line_perm = np.zeros((N, N))
    
    for node1, (r1, c1) in enumerate(edge_index_g1):
        for node2, (r2, c2) in enumerate(edge_index_g2):
            
            
            if (perm[r1, r2] and perm[c1, c2]) or (perm[r1, c2] and perm[c1, r2]):
                line_correspond.append([node1, node2])
                line_perm[node1, node2] = 1 

    # print('line_correspond: ', line_correspond)
    # print('line_perm: ',line_perm)

    # print('line graph correspond number: ', len(line_correspond))
    return line_perm