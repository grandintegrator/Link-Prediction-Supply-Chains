import networkx as nx
import logging
import dgl
import pickle
import torch
import pandas as pd
import numpy as np
import warnings
from dataset import KnowledgeGraphGenerator
from dgl.data import DGLDataset
from typing import List, Dict, Any

# :)
warnings.filterwarnings("ignore")

# Logger preferences
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig()

# Pandas debugging options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class SupplyKnowledgeGraphDataset(DGLDataset):
    def __init__(self, path: str = 'data/02_intermediate/',
                 from_scratch: bool = True):
        """Class creates a DGLDataset object for a multi-graph to be later
        used with DGL for Link prediction/classification.
        """
        ########################################################################
        # Load dataset from dataset.py
        # Spits an object with all sanitised data (with nice lower names, etc.)
        ########################################################################
        dataset_generator = KnowledgeGraphGenerator()
        dataset = dataset_generator.load(from_scratch=from_scratch,
                                         path=path)
        if from_scratch:
            logger.info('Loaded graphs with the following dimensions:')
            logger.info('====================================================')
            logger.info(f'cG has {len(dataset.cG_clean.edges)} edges')
            logger.info(f'bG has {len(dataset.bG_clean.edges)} edges')
            logger.info(f'G has have {len(dataset.G_clean.edges)} edges')
            logger.info(f'capability_product_graph has {len(dataset.capability_product_graph.edges)} edges')
            logger.info(f'company_capability_graph has {len(dataset.company_capability_graph.edges)} edges')
            logger.info(f'{len(dataset.capabilities_all)} Capabilities loaded')
            logger.info(f'{len(dataset.processes_all)} Processes loaded')
            logger.info(f'{len(dataset.companies_all)} Companies loaded')
            logger.info('====================================================')

        self.bG = dataset.bG_clean
        self.company_capability_graph = dataset.company_capability_graph
        self.capability_product_graph = dataset.capability_product_graph
        self.capability_graph = dataset.capability_graph
        self.G = dataset.G_clean
        self.companies_all = dataset.companies_all
        self.processes_all = list(dataset.processes_all)
        self.capabilities_all = list(dataset.capabilities_all)
        self.cG = dataset.cG_clean

        # Declare empty requirements for GNN libraries
        self.company_nodes = None
        self.process_nodes = None
        self.capability_nodes = None
        self.triplets = None
        self.graph = None
        self.num_rels = None
        self.train_graph = None
        self.valid_graph = None

        logger.info('All graphs loaded to memory - moving to process...')
        super().__init__(name='supply_knowledge_graph')

    def create_nodes_data(self) -> None:
        """Function creates a dataframe for all nodes that maps the Name to
        an ID and assigns a type to the node as well.

        DataFrame = NODE_NAME | NODE_TYPE | NODE_ID
        """
        # Add the company nodes
        self.company_nodes = \
            pd.DataFrame({'NODE_NAME': self.companies_all,
                          'NODE_TYPE': 'COMPANY'})
        # Create unique company IDs
        self.company_nodes['NODE_ID'] = self.company_nodes.index.astype('int')

        # Add in the process nodes
        # process_nodes_names = list(set(([el[0] for el in self.bG.edges])))
        self.process_nodes = \
            pd.DataFrame({'NODE_NAME': self.processes_all,
                          'NODE_TYPE': 'PROCESS'})
        self.process_nodes['NODE_ID'] = self.process_nodes.index.astype('int')

        # Same process for all Capability Nodes
        self.capability_nodes = \
            pd.DataFrame({'NODE_NAME': self.capabilities_all,
                          'NODE_TYPE': 'CAPABILITY'})
        self.capability_nodes['NODE_ID'] = \
            self.capability_nodes.index.astype('int')

    logger.info('All nodes & IDs have been added to memory')

    def create_triples(self) -> pd.DataFrame:
        """Function uses the bG and G graphs within the graph_object
        to create a multi relational dataframe

         |src| src_id| | dst| dst_id | relation_type | src_type | dst_type|

        Returns:
            Dataframe containing knowledge graph for supply chains
        """

        # This generates the node IDs in self
        self.create_nodes_data()

        ########################################################################
        # Create lookup dictionary for all nodes (SEQUENTIAL FOR ALL NODES)
        ########################################################################
        all_nodes_frame = pd.concat([self.company_nodes,
                                     self.process_nodes,
                                     self.capability_nodes
                                     ], ignore_index=True)

        all_nodes_frame.reset_index(drop=True, inplace=True)
        all_nodes_frame['NODE_ID'] = all_nodes_frame.index.astype('int')

        node_lookup_map = (
            all_nodes_frame[['NODE_NAME',
                             'NODE_ID']].set_index('NODE_NAME')
                .to_dict()['NODE_ID']
        )

        del all_nodes_frame

        def sub_frame_generator(original_graph: nx.DiGraph,
                                relation_type: str,
                                src_type: str,
                                dst_type: str) -> pd.DataFrame:
            """
            Args:
                original_graph: Graph generated from dataset
                relation_type: Relationship type, e.g. 'buys_from'
                src_type: 'Company', 'Product', or 'Capability'
                dst_type: 'Company', 'Product', or 'Capability'

            Returns:
                Dataframe -

              src| src_id| | dst| dst_id | relation_type | src_type | dst_type
            """
            sources_targets_nx = nx.to_pandas_edgelist(original_graph)

            empty_data_frame = pd.DataFrame({'src': [],
                                             'dst': [],
                                             'src_id': [],
                                             'dst_id': []})
            empty_data_frame['src'] = sources_targets_nx['source']
            empty_data_frame['dst'] = sources_targets_nx['target']
            empty_data_frame['src_id'] = \
                empty_data_frame['src'].map(node_lookup_map)
            empty_data_frame['dst_id'] = \
                empty_data_frame['dst'].map(node_lookup_map)

            empty_data_frame['relation_type'] = relation_type
            empty_data_frame['src_type'] = src_type
            empty_data_frame['dst_type'] = dst_type
            return empty_data_frame

        ########################################################################
        # SUB-FRAME (Company -> buys_from -> Company)
        ########################################################################
        companies_company_frame = sub_frame_generator(self.G,
                                                      relation_type='buys_from',
                                                      src_type='Company',
                                                      dst_type='Company')

        ########################################################################
        # SUB-FRAME (Company -> company_makes -> Product)
        ########################################################################
        company_product_frame = (
            sub_frame_generator(self.bG,
                                relation_type='company_makes',
                                src_type='Company',
                                dst_type='Product')
        )

        # products_relations_frame = pd.DataFrame({'src': [],
        #                                          'dst': [],
        #                                          'src_id': [],
        #                                          'dst_id': []})
        #
        # sources_targets = nx.to_pandas_edgelist(self.bG)
        # products_relations_frame['src'] = sources_targets['source']
        # products_relations_frame['dst'] = sources_targets['target']
        #
        # # COND_1 return: (1407, 2) - would expect this to be 0
        # cond_2 = products_relations_frame['dst'].isin(self.G.nodes)
        # # COND_2 return: (110989, 2) - would expect this to be 203420
        # products_relations_frame = products_relations_frame.loc[cond_2, :]
        #
        # products_relations_frame['src_id'] = \
        #     products_relations_frame['src'].map(node_lookup_products)
        #
        # products_relations_frame['dst_id'] = \
        #     products_relations_frame['dst'].map(node_lookup_products)
        #
        # products_relations_frame['relation_type'] = 'company_makes'
        # products_relations_frame['subject_type'] = 'Process'
        # products_relations_frame['object_type'] = 'Company'
        # del sources_targets, cond_2

        ########################################################################
        # SUB-FRAME (Product -> complimentary_product_to -> Product)
        ########################################################################
        product_product_frame = (
            sub_frame_generator(self.cG,
                                relation_type='complimentary_product_to',
                                src_type='Product',
                                dst_type='Product')
        )

        ########################################################################
        # SUB-FRAME (Capability -> capability_produces -> Product)
        ########################################################################
        capability_product_frame = (
            sub_frame_generator(self.capability_product_graph,
                                relation_type='capability_produces',
                                src_type='Capability',
                                dst_type='Product')
        )

        ########################################################################
        # SUB-FRAME (Company -> has_capability -> Capability)
        ########################################################################
        company_capability_frame = (
            sub_frame_generator(self.company_capability_graph,
                                relation_type='has_capability',
                                src_type='Company',
                                dst_type='Capability')
        )

        ########################################################################
        # Add everything together into a triplets frame
        ########################################################################
        self.triplets = pd.concat([companies_company_frame,
                                   product_product_frame,
                                   capability_product_frame,
                                   company_product_frame,
                                   company_capability_frame],
                                  ignore_index=True)

        # self.triplets.shape = (796124, 7)
        del companies_company_frame, product_product_frame,\
            capability_product_frame, company_capability_frame

        self.triplets = (
            self.triplets.reset_index(drop=True)
            .dropna()
        )

        self.triplets['src_id'] = self.triplets['src_id'].astype('int')
        self.triplets['dst_id'] = self.triplets['dst_id'].astype('int')

        self.triplets['rel_id'] = (
            self.triplets['relation_type'].map({'buys_from': 1,
                                                'company_makes': 2,
                                                'complimentary_product_to': 3,
                                                'capability_produces': 4,
                                                'has_capability': 5})
        )

        logger.info('Triplets created for all entities in the KG')
        return self.triplets

    def process(self) -> None:
        """
        super method -> Gets called first.
        """
        self.create_triples()
        logger.info('Triplets created. Starting processing to pytorch...')

        ########################################################################
        # Create Heterograph from DGL - This process is done iteratively
        ########################################################################
        cond = self.triplets['relation_type'] == 'buys_from'
        buys_from = self.triplets.loc[cond]
        company_buying_triples = \
            [torch.tensor((int(src_id), int(dst_id)),
                          dtype=torch.int32) for src_id, dst_id
             in zip(buys_from.src_id, buys_from.dst_id)]

        cond = self.triplets['relation_type'] == 'company_makes'
        makes_product = self.triplets.loc[cond]
        makes_product_triples = \
            [torch.tensor((int(src_id), int(dst_id)),
                          dtype=torch.int32) for src_id, dst_id
             in zip(makes_product.src_id, makes_product.dst_id)]

        cond = self.triplets['relation_type'] == 'complimentary_product_to'
        product_product = self.triplets.loc[cond]
        product_product_triples = \
            [torch.tensor((int(src_id), int(dst_id)),
                          dtype=torch.int32) for src_id, dst_id
             in zip(product_product.src_id, product_product.dst_id)]

        cond = self.triplets['relation_type'] == 'capability_produces'
        capability_product = self.triplets.loc[cond]
        capability_product_triples = \
            [torch.tensor((int(src_id), int(dst_id)),
                          dtype=torch.int32) for src_id, dst_id
             in zip(capability_product.src_id, capability_product.dst_id)]

        cond = self.triplets['relation_type'] == 'has_capability'
        company_capability = self.triplets.loc[cond]
        company_capability_triples = \
            [torch.tensor((int(src_id), int(dst_id)),
                          dtype=torch.int32) for src_id, dst_id
             in zip(company_capability.src_id, company_capability.dst_id)]

        del makes_product, product_product, buys_from, cond, self.triplets
        data_dict = {
            ('company', 'buys_from', 'company'): company_buying_triples,
            ('company', 'makes_product', 'product'): makes_product_triples,
            ('product', 'complimentary_product_to', 'product'): product_product_triples,
            ('capability', 'capability_produces', 'product'): capability_product_triples,
            ('company', 'has_capability', 'capability'): company_capability_triples
        }

        self.graph = dgl.heterograph(data_dict)
        self.num_rels = len(self.graph.etypes)

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


loader = SupplyKnowledgeGraphDataset()
data_frame = loader[0]


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
            batch_size=self.params.testing.batch_size,
            shuffle=True,
            drop_last=False,
            # pin_memory=True,
            num_workers=self.params.modelling.num_workers)
        return test_data_loader

