import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import pandas as pd
import dgl.function as fn
import networkx as nx

import pickle
import dgl.data
import logging
import dgl
import itertools
from tqdm import tqdm

# Define a Heterograph Conv model
import dgl.nn as dglnn
import torch
from sklearn.metrics import roc_auc_score
from ingestion.dataloader import SupplyKnowledgeGraphDataset

loader = SupplyKnowledgeGraphDataset()
data_sp = loader[0]
n_hetero_features = 2

# n_users = 1000
# n_items = 500
# n_follows = 3000
# n_clicks = 5000
# n_dislikes = 500
# n_hetero_features = 10
# n_user_classes = 5
# n_max_clicks = 10
#
# follow_src = np.random.randint(0, n_users, n_follows)
# follow_dst = np.random.randint(0, n_users, n_follows)
# click_src = np.random.randint(0, n_users, n_clicks)
# click_dst = np.random.randint(0, n_items, n_clicks)
# dislike_src = np.random.randint(0, n_users, n_dislikes)
# dislike_dst = np.random.randint(0, n_items, n_dislikes)
#
# hetero_graph = dgl.heterograph({
#     ('user', 'follow', 'user'): (follow_src, follow_dst),
#     ('user', 'followed-by', 'user'): (follow_dst, follow_src),
#     ('user', 'click', 'item'): (click_src, click_dst),
#     ('item', 'clicked-by', 'user'): (click_dst, click_src),
#     ('user', 'dislike', 'item'): (dislike_src, dislike_dst),
#     ('item', 'disliked-by', 'user'): (dislike_dst, dislike_src)})
#
#
#
# hetero_graph.nodes['user'].data['feature'] = torch.randn(n_users,
#                                                          n_hetero_features)
# hetero_graph.nodes['item'].data['feature'] = torch.randn(n_items,
#                                                          n_hetero_features)

hetero_graph = data_sp
n_companies = hetero_graph.num_nodes('company')
n_products = hetero_graph.num_nodes('product')

hetero_graph.nodes['company'].data['feature'] = torch.randn(n_companies,
                                                            n_hetero_features)
hetero_graph.nodes['product'].data['feature'] = torch.randn(n_products,
                                                            n_hetero_features)

# hetero_graph.nodes['user'].data['label'] = torch.randint(0, n_user_classes, (n_users,))
# hetero_graph.edges['click'].data['label'] = torch.randint(1, n_max_clicks, (n_clicks,)).float()
# randomly generate training masks on user nodes and click edges
# hetero_graph.nodes['user'].data['train_mask'] = torch.zeros(n_users, dtype=torch.bool).bernoulli(0.6)
# hetero_graph.edges['click'].data['train_mask'] = torch.zeros(n_clicks, dtype=torch.bool).bernoulli(0.6)


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    # neg_src_check = src.expand(1, src.shape[0]*k).flatten(-1, 0)
    neg_dst = torch.randint(0, graph.number_of_nodes(vtype), (len(src) * k,))
    #
    # # Split edge set for training and testing
    # u, v = graph.edges(etype=etype)
    #
    # eids = np.arange(graph.number_of_edges(etype))
    # eids = np.random.permutation(eids)
    # test_size = int(len(eids) * 0.1)
    # train_size = graph.number_of_edges(etype) - test_size
    # test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    # train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
    #
    # # Find all negative edges and split them for training and testing
    # adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    # adj_neg = 1 - adj.todense() - np.eye(graph.number_of_nodes())
    # neg_u, neg_v = np.where(adj_neg != 0)
    #
    # neg_eids = np.random.choice(len(neg_u), graph.number_of_edges(etype))
    # test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    # train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
    #
    #
    # train_pos_g = dgl.heterograph({etype: (train_pos_u, train_pos_v)},
    #                                  num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})
    #
    # train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    # train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
    #
    # test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    # test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()

    def forward(self, g, neg_g, x, etype):
        h = self.sage(g, x)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)


def compute_loss(pos_score, neg_score):
    # Margin loss
    n_edges = pos_score.shape[0]
    return (1 - neg_score.view(n_edges, -1) + pos_score.unsqueeze(1)).clamp(min=0).mean()


hetero_graph = data_sp
edge = ('company', 'buys_from', 'company')
# edge_old = ('user', 'click', 'item')

k = 1
n_hetero_features = 2
link_predict = True
out_features = 2 if link_predict else 0
model = Model(n_hetero_features, 20, out_features, hetero_graph.etypes)
# user_feats = hetero_graph.nodes['user'].data['feature']
# item_feats = hetero_graph.nodes['item'].data['feature']

company_feats = hetero_graph.nodes['company'].data['feature']
product_feats = hetero_graph.nodes['product'].data['feature']

node_features = {'company': company_feats, 'product': product_feats}
opt = torch.optim.Adam(model.parameters())


for epoch in tqdm(range(1000)):
    negative_graph = construct_negative_graph(hetero_graph, k, edge)
    pos_score, neg_score = model(hetero_graph,
                                 negative_graph,
                                 node_features,
                                 edge)
    loss = compute_loss(pos_score, neg_score)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    negative_graph = construct_negative_graph(hetero_graph, k, edge)
    pos_score, neg_score = model(hetero_graph, negative_graph, node_features, edge)
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    print(roc_auc_score(labels, scores))
