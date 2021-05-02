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
import torch
import itertools
from tqdm import tqdm
from dgl.nn import SAGEConv

# Pandas debugging options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Logger preferences
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig()


################################################################################
# LOAD DATASET FROM NETWORKX
################################################################################
class DataLoaderDGL(object):
    def __init__(self, path: str = '../data/02_intermediate/marklinesEdges.p'):
        """Loads NetworkX graph from pickle dataset into a DGL Graph Object
        Args:
            path: Location of the MarkLines pickled object
        """
        logger.info('Loading NetworkX graph object into DGL')
        # load existing graph object
        self.multi_pair_frame = None
        self.graph_object = pickle.load(open(path, "rb"))
        logger.info('NetworkX graph loaded')

    def get_dgl_graph(self):
        return dgl.from_networkx(self.graph_object.G,
                                 edge_attrs=None)


data_loader = DataLoaderDGL()
dataset = data_loader.get_dgl_graph()
g = dataset

EMBEDDING_INITIALISED = 15
embed = nn.Embedding(g.number_of_nodes(), EMBEDDING_INITIALISED)
g.ndata['feat'] = embed.weight

################################################################################
# Prepare training and testing sets
################################################################################
# Split edge set for training and testing
u, v = g.edges()

eids = np.arange(g.number_of_edges())
eids = np.random.permutation(eids)
test_size = int(len(eids) * 0.1)
train_size = g.number_of_edges() - test_size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

# Find all negative edges and split them for training and testing
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
neg_u, neg_v = np.where(adj_neg != 0)

neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)
test_neg_u, test_neg_v = (
    neg_u[neg_eids[:test_size]],
    neg_v[neg_eids[:test_size]]
)
train_neg_u, train_neg_v = (
    neg_u[neg_eids[test_size:]],
    neg_v[neg_eids[test_size:]]
)

train_g = dgl.remove_edges(g, eids[:test_size])
test_g = dgl.remove_edges(g, eids[train_size:])

################################################################################
# MODEL - Two layers of GraphSAGEConv
################################################################################
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
        self.conv3 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.leaky_relu(h)
        h = self.conv2(g, h)
        h = F.leaky_relu(h)
        h = self.conv3(g, h)
        return h


################################################################################
# Create positive and negative graphs from the sets of nodes for easier compute
################################################################################
train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())


################################################################################
# The benefit of treating the pairs of nodes as a graph is that you can
# use the ``DGLGraph.apply_edges`` method, which conveniently computes new
# # edge features based on the incident nodes’ features and the original
# # edge features (if applicable).
#
# DGL provides a set of optimized builtin functions to compute new
# edge features based on the original node/edge features. For example,
# ``dgl.function.u_dot_v`` computes a dot product of the incident nodes’
# representations for each edge.
################################################################################
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


######################################################################
# You can also write your own function if it is complex.
# For instance, the following module produces a scalar score on each edge
# by concatenating the incident nodes’ features and passing it to an MLP.

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']


model = GraphSAGE(train_g.ndata['feat'].shape[1], 16)
pred = DotPredictor()


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc_ap(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return [roc_auc_score(labels, scores),
            average_precision_score(labels, scores)]


optimizer = torch.optim.Adam(itertools.chain(model.parameters(),
                                             pred.parameters()), lr=0.01)

all_logits = []
for e in range(2000):
    # forward
    h = model(train_g, train_g.ndata['feat'])
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    if e % 5 == 0:
        logger.info('In epoch {}, loss: {}'.format(e, loss))

# ----------- 5. check results ------------------------ #
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    auc, av_precision = compute_auc_ap(pos_score, neg_score)
    logger.info(f'Average AUC at Final: {auc}')
    logger.info(f'Average Precision: {av_precision}')
    # print('AUC at final Epoch', compute_auc_ap(pos_score, neg_score))
