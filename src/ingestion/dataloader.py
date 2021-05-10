import networkx as nx
import logging
import dgl
import pickle

from dgl.data import DGLDataset
from ingestion.utils import (
    get_adj_and_degrees,
    sample_edge_neighborhood
)

import torch
import pandas as pd
import warnings
from typing import List

# :)
warnings.filterwarnings("ignore")

# Logger preferences
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig()


class SupplyKnowledgeGraphDataset(DGLDataset):
    def __init__(self, path: str = 'data/02_intermediate/'):
        with open(path + 'G.pickle', 'rb') as f:
            self.G = pickle.load(f)
        with open(path + 'bG.pickle', 'rb') as f:
            self.bG = pickle.load(f)
        with open(path + 'cG.pickle', 'rb') as f:
            self.cG = pickle.load(f)

        self.company_nodes = None
        self.process_nodes = None
        self.triplets = None
        self.graph = None
        self.num_rels = None
        self.train_graph = None
        self.valid_graph = None

        logger.info('All graphs loaded to memory')
        super().__init__(name='supply_knowledge_graph')

    def create_nodes_data(self) -> None:
        # Add the company nodes
        self.company_nodes = pd.DataFrame({'NODE_NAME': self.G.nodes,
                                           'NODE_TYPE': 'COMPANY'})
        # Create unique company IDs
        self.company_nodes['NODE_ID'] = self.company_nodes.index.astype('int')

        # Add in the process nodes
        process_nodes_names = list(set(([el[0] for el in self.bG.edges])))
        self.process_nodes = pd.DataFrame({'NODE_NAME': process_nodes_names,
                                           'NODE_TYPE': 'PROCESS'})
        self.process_nodes['NODE_ID'] = self.process_nodes.index.astype('int')
        logger.info('All nodes & IDs have been added to memory')

    def create_triples(self, index_all_nodes: bool = True) -> pd.DataFrame:
        """Function uses the bG and G graphs within the graph_object
        to create a multi relational dataframe

         |src| src_id| | dst| dst_id | relation_type | src_type | dst_type|

        Returns:
            Dataframe containing knowledge graph for supply chains
        """
        # This generates the node IDs in self
        self.create_nodes_data()
        ########################################################################
        # Create buy-sell company-company sub-frame
        ########################################################################
        companies_relations_frame = pd.DataFrame({'src': [],
                                                  'dst': [],
                                                  'src_id': [],
                                                  'dst_id': []})

        sources_targets_companies = nx.to_pandas_edgelist(self.G)
        companies_relations_frame['src'] = sources_targets_companies['source']
        companies_relations_frame['dst'] = sources_targets_companies['target']

        if index_all_nodes:
            all_nodes_frame = pd.concat([self.company_nodes,
                                         self.process_nodes,
                                         ], ignore_index=True)

            all_nodes_frame.reset_index(drop=True, inplace=True)
            all_nodes_frame['NODE_ID'] = all_nodes_frame.index.astype('int')

            node_lookup_companies = (
                all_nodes_frame[['NODE_NAME', 'NODE_ID']]
                    .set_index('NODE_NAME').to_dict()['NODE_ID']
            )
            node_lookup_products = (
                all_nodes_frame[['NODE_NAME', 'NODE_ID']]
                    .set_index('NODE_NAME').to_dict()['NODE_ID']
            )

        else:
            node_lookup_companies = (
                self.company_nodes[['NODE_NAME', 'NODE_ID']]
                    .set_index('NODE_NAME').to_dict()['NODE_ID']
            )
            node_lookup_products = (
                self.process_nodes[['NODE_NAME', 'NODE_ID']]
                    .set_index('NODE_NAME').to_dict()['NODE_ID']
            )

        companies_relations_frame['src_id'] = \
            companies_relations_frame['src'].map(node_lookup_companies)

        companies_relations_frame['dst_id'] = \
            companies_relations_frame['dst'].map(node_lookup_companies)

        companies_relations_frame['relation_type'] = 'buys_from'
        companies_relations_frame['subject_type'] = 'Company'
        companies_relations_frame['object_type'] = 'Company'

        del sources_targets_companies
        ########################################################################
        # Create product_company sub-frame
        ########################################################################
        products_relations_frame = pd.DataFrame({'src': [],
                                                 'dst': [],
                                                 'src_id': [],
                                                 'dst_id': []})

        sources_targets = nx.to_pandas_edgelist(self.bG)
        products_relations_frame['src'] = sources_targets['source']
        products_relations_frame['dst'] = sources_targets['target']

        # products_relations_frame.shape = 203420 rows x 2 columns
        # TODO: Check the following logic with Edward
        cond_1 = products_relations_frame['src'].isin(self.G.nodes)
        # COND_1 return: (1407, 2) - would expect this to be 0
        cond_2 = products_relations_frame['dst'].isin(self.G.nodes)
        # COND_2 return: (110989, 2) - would expect this to be 203420
        products_relations_frame = products_relations_frame.loc[cond_2, :]

        products_relations_frame['src_id'] = \
            products_relations_frame['src'].map(node_lookup_products)

        products_relations_frame['dst_id'] = \
            products_relations_frame['dst'].map(node_lookup_products)

        products_relations_frame['relation_type'] = 'makes_product'
        products_relations_frame['subject_type'] = 'Process'
        products_relations_frame['object_type'] = 'Company'
        del sources_targets, cond_1, cond_2

        ########################################################################
        # Create product-product sub frame - like protein-protein network
        ########################################################################
        process_process_frame = pd.DataFrame({'src': [],
                                              'dst': [],
                                              'src_id': [],
                                              'dst_id': []})

        sources_targets = nx.to_pandas_edgelist(self.cG)
        process_process_frame['src'] = sources_targets['source']
        process_process_frame['dst'] = sources_targets['target']

        process_process_frame['src_id'] = \
            process_process_frame['src'].map(node_lookup_products)

        process_process_frame['dst_id'] = \
            process_process_frame['dst'].map(node_lookup_products)

        process_process_frame['relation_type'] = 'product_product'
        process_process_frame['subject_type'] = 'Process'
        process_process_frame['object_type'] = 'Process'
        del sources_targets

        ########################################################################
        # Add everything together into a triplets frame
        ########################################################################
        self.triplets = pd.concat([companies_relations_frame,
                                   products_relations_frame,
                                   process_process_frame],
                                  ignore_index=True)
        del companies_relations_frame, products_relations_frame,\
            process_process_frame
        self.triplets = (
            self.triplets.reset_index(drop=True)
            .dropna()
        )

        self.triplets['src_id'] = self.triplets['src_id'].astype('int')
        self.triplets['dst_id'] = self.triplets['dst_id'].astype('int')

        self.triplets['rel_id'] = (
            self.triplets['relation_type'].map({'buys_from': 1,
                                                'makes_product': 2,
                                                'product_product': 3})
        )

        logger.info('Triplets created for all entities in the KG')
        return self.triplets

    # def get_triplets_rcgn_link_prediction(self) -> List:
    #     """Creates a type of relation list that rgcn_link_prediction.py
    #     can handle
    #     """
    #     self.triplets = self.create_triples(index_all_nodes=True)
    #     return [[src_id, rel_id, dst_id] for
    #             src_id, rel_id, dst_id in
    #             zip(self.triplets.src_id,
    #                 self.triplets.rel_id,
    #                 self.triplets.dst_id)]
    #
    # def get_train_valid_test_graph(self,
    #                                graph_batch_size: int = 30000,
    #                                graph_split_size: float = 0.5,
    #                                negative_sample: int = 10,
    #                                edge_sampler: str = 'neighbor'):
    #
    #     triplets = self.get_triplets_rcgn_link_prediction()
    #
    #     adj_list, degrees = get_adj_and_degrees(self.graph.num_nodes(),
    #                                             triplets)
    #     # triplets = tuple(triplets)
    #     logger.info('=========================================================')
    #     logger.info('Sampling from Edge Neighbourhood')
    #     edges = sample_edge_neighborhood(adj_list=adj_list,
    #                                      degrees=degrees,
    #                                      n_triplets=len(triplets),
    #                                      sample_size=int(0.1*len(triplets)))
    #
    #     test_triplets = [triplets[i] for i in edges]
    #     train_indices = list(set(range(len(triplets))) - set(edges))
    #     train_triplets = [triplets[i] for i in train_indices]
    #     return train_triplets, test_triplets

    def process(self):
        self.create_triples()
        logger.info('Triplets created. Starting processing to pytorch...')
        ########################################################################
        # Create Heterograph from DGL
        ########################################################################
        cond = self.triplets['relation_type'] == 'buys_from'
        buys_from = self.triplets.loc[cond]
        company_buying_triples = \
            [torch.tensor((int(src_id), int(dst_id)),
                          dtype=torch.int32) for src_id, dst_id
             in zip(buys_from.src_id, buys_from.dst_id)]

        cond = self.triplets['relation_type'] == 'makes_product'
        makes_product = self.triplets.loc[cond]
        makes_product_triples = \
            [torch.tensor((int(src_id), int(dst_id)),
                          dtype=torch.int32) for src_id, dst_id
             in zip(makes_product.src_id, makes_product.dst_id)]

        cond = self.triplets['relation_type'] == 'product_product'
        product_product = self.triplets.loc[cond]
        product_product_triples = \
            [torch.tensor((int(src_id), int(dst_id)),
                          dtype=torch.int32) for src_id, dst_id
             in zip(product_product.src_id, product_product.dst_id)]

        del makes_product, product_product, buys_from, cond, self.triplets
        data_dict = {
            ('company', 'buys_from', 'company'): company_buying_triples,
            ('company', 'makes_product', 'product'): makes_product_triples,
            ('product', 'product_product', 'product'): product_product_triples
        }
        del company_buying_triples, makes_product_triples, product_product_triples
        self.graph = dgl.heterograph(data_dict)
        self.num_rels = len(self.graph.etypes)
        # self.train_graph, self.valid_graph = self.get_train_valid_test_graph()

    # def construct_negative_graph(self,
    #                              k: int,
    #                              etype: tuple = ('company',
    #                                              'buys_from',
    #                                              'company'))\
    #         -> dgl.DGLHeteroGraph:
    #     utype, _, vtype = etype
    #     src, dst = self.graph.edges(etype=etype)
    #     neg_src = src.repeat_interleave(k)
    #     neg_dst = torch.randint(0, self.graph.num_nodes(vtype), (len(src) * k,))
    #     return dgl.heterograph(
    #         {etype: (neg_src, neg_dst)},
    #         num_nodes_dict={ntype: self.graph.num_nodes(ntype)
    #                         for ntype in self.graph.ntypes})

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


class SCDataLoader(object):
    def __init__(self, params):
        self.params = params

        loader = SupplyKnowledgeGraphDataset()
        self.full_graph = loader[0]
        self.edge_types = self.full_graph.etypes
        self.training_data = None
        self.testing_data = None

    def get_training_testing(self) -> None:
        """
        # TODO: Add in validation split too - not just train, test
        """
        # randomly generate training masks for our buys_from edges
        # Need to make sure this is reproducible.
        buys_from_train_ids = \
            torch.zeros(self.full_graph.number_of_edges('buys_from'),
                        dtype=torch.bool).bernoulli(1-self.params.modelling.test_p)

        # Get all of the edges in the company - buys_from - company edges
        src, dst = self.full_graph.edges(etype='buys_from')

        # Split them into train and test based on the Bernoulli IDs
        src_train = src[buys_from_train_ids]
        dst_train = dst[buys_from_train_ids]

        src_test = src[~buys_from_train_ids]
        dst_test = src[~buys_from_train_ids]

        # Create TRAIN and TEST data dictionaries as unique heterographs
        edge_type_1 = ('company', 'buys_from', 'company')
        edge_type_2 = ('company', 'makes_product', 'product')
        edge_type_3 = ('product', 'product_product', 'product')

        train_data_dict = {
            edge_type_1: (src_train, dst_train),
            edge_type_2: self.full_graph.edges(etype='makes_product'),
            edge_type_3: self.full_graph.edges(etype='product_product')
        }

        test_data_dict = {
            edge_type_1: (src_test, dst_test),
            edge_type_2: self.full_graph.edges(etype='makes_product'),
            edge_type_3: self.full_graph.edges(etype='product_product')
        }

        self.training_data = dgl.heterograph(train_data_dict)
        self.testing_data = dgl.heterograph(test_data_dict)

    def get_training_dataloader(self) -> dgl.dataloading.EdgeDataLoader:
        # Create the sampler object
        self.get_training_testing()
        n_companies = self.training_data.num_nodes('company')
        n_products = self.training_data.num_nodes('product')
        n_hetero_features = self.params.modelling.num_node_features

        # Initialise the training data features
        self.training_data.nodes['company'].data['feature'] = (
            torch.randn(n_companies, n_hetero_features)
        )

        self.training_data.nodes['product'].data['feature'] = (
            torch.randn(n_products, n_hetero_features)
        )
        graph_eid_dict = \
            {etype: self.training_data.edges(etype=etype, form='eid')
             for etype in self.training_data.etypes}

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        # sampler = (
        #     dgl.dataloading.MultiLayerNeighborSampler(2)
        # )
        negative_sampler = dgl.dataloading.negative_sampler.Uniform(10)

        train_data_loader = dgl.dataloading.EdgeDataLoader(
            self.training_data, graph_eid_dict, sampler,
            negative_sampler=negative_sampler,
            batch_size=self.params.modelling.batch_size,
            shuffle=True,
            drop_last=False,
            # pin_memory=True,
            num_workers=self.params.modelling.num_workers)
        return train_data_loader

    def get_test_data_loader(self) -> dgl.dataloading.EdgeDataLoader:
        # Creates testing data loader for evaluation
        self.get_training_testing()
        n_companies = self.testing_data.num_nodes('company')
        n_products = self.testing_data.num_nodes('product')
        n_hetero_features = self.params.modelling.num_node_features

        # Initialise the training data features
        self.testing_data.nodes['company'].data['feature'] = (
            torch.randn(n_companies, n_hetero_features)
        )

        self.testing_data.nodes['product'].data['feature'] = (
            torch.randn(n_products, n_hetero_features)
        )
        graph_eid_dict = \
            {etype: self.testing_data.edges(etype=etype, form='eid')
             for etype in self.testing_data.etypes}

        # sampler = dgl.dataloading.MultiLayerFullNeighborSampler([30, 30])
        # sampler = (
        #     dgl.dataloading.MultiLayerNeighborSampler([60, 60])
        # )

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        negative_sampler = dgl.dataloading.negative_sampler.Uniform(10)

        test_data_loader = dgl.dataloading.EdgeDataLoader(
            self.testing_data, graph_eid_dict, sampler,
            negative_sampler=negative_sampler,
            batch_size=self.params.modelling.batch_size,
            shuffle=True,
            drop_last=False,
            # pin_memory=True,
            num_workers=self.params.modelling.num_workers)
        return test_data_loader

