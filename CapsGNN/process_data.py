import networkx as nx
import os
import pickle as pkl
import json as js
import jsbeautifier as jsb
import sys
import scipy
import numpy as np
import scipy.sparse as sp
import torch

sys.path.append("..")
from const import *

graph_path = GRAPH_PATH
files = os.listdir('../'+graph_path)


def reg_graph(G):
    return nx.from_scipy_sparse_matrix(nx.to_scipy_sparse_matrix(G))


def is_idx_from_zero(G):
    last_node = sorted(G.nodes)[-1]
    return type(last_node) == int and last_node == len(G) - 1


def get_features(G):
    adj = nx.adjacency_matrix(G)
    node_num = adj.shape[0]
    adj_ = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj_.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).toarray()
    _, features = scipy.linalg.eigh(adj_normalized, eigvals=(node_num - feature_dim, node_num - 1))
    features = torch.FloatTensor(features)
    return features


if __name__ == '__main__':
    for file in files:
        graphs = []
        graph_names = []
        graph_labels = []
        if not os.path.isdir(file):
            if 'pkl' in file:
                with open(graph_path + "/" + file, 'rb') as f:
                    graphs = pkl.load(f, encoding='latin1')
                    os.makedirs(graph_path + "/" + file, exist_ok=True)
                    graph_names = [file + f'_{str(i)}' for i in range(len(graphs))]
                    graph_labels = [g.graph['label'] for g in graphs]
                    graph_features = [get_features(g) for g in graphs]
                for i in range(len(graphs)):
                    if is_idx_from_zero(graphs[i]):
                        continue
                    else:
                        graphs[i] = reg_graph(graphs[i])
                for g, name, feature in zip(graphs, graph_names, graph_features):
                    g = nx.from_numpy_matrix(nx.to_numpy_matrix(g))
                    g_nodes = {}
                    for node in g.nodes():
                        g_nodes[str(node)] = feature
                    with open(graph_path + "/" + file + '/' + name + '.json', 'w') as f:
                        f.write(
                            jsb.beautify(js.dumps(
                                {'edges': [list(i) for i in list(g.edges())], 'labels': g_nodes, 'target': g.graph['label']})))
