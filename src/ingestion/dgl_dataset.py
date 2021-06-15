import networkx as nx
import logging
import dgl
import os
import torch
import pandas as pd
import numpy as np
import warnings
from ingestion.dataset import KnowledgeGraphGenerator
from dgl.data import DGLDataset
from pprint import pformat
from typing import Dict, Any

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
    def __init__(self, params,
                 path: str = '../data/02_intermediate/',
                 from_scratch: bool = False,
                 triplets_from_scratch: bool = False,
                 load_graph: bool = True):
        """Class creates a DGLDataset object for a multi-graph to be later
        used with DGL for Link prediction/classification.
        """
        ########################################################################
        # Load dataset from dataset.py
        # Spits an object with all sanitised data (with nice lower names, etc.)
        ########################################################################
        self.triplets_from_scratch = params.triplets_from_scratch
        self.load_graph = params.load_graph
        self.data_path = path

        dataset_generator = KnowledgeGraphGenerator(params=params,
                                                    path=self.data_path)
        dataset = dataset_generator.load(from_scratch=params.from_scratch,
                                         path=self.data_path)
        if params.from_scratch:
            logger.info('Loaded graphs with the following dimensions:')
            logger.info('====================================================')
            logger.info(f'cG has {len(dataset.cG_clean.edges)} edges')
            logger.info(f'bG has {len(dataset.bG_clean.edges)} edges')
            logger.info(f'G has have {len(dataset.G_clean.edges)} edges')
            logger.info(f'capability_product_graph has {len(dataset.capability_product_graph.edges)} edges')
            logger.info(f'company_capability_graph has {len(dataset.company_capability_graph.edges)} edges')
            logger.info(f'company_capability_graph has {len(dataset.company_country_graph.edges)} edges')
            logger.info(f'company_capability_graph has {len(dataset.company_certification_graph.edges)} edges')
            logger.info(f'{len(dataset.capabilities_all)} Capabilities loaded')
            logger.info(f'{len(dataset.processes_all)} Processes loaded')
            logger.info(f'{len(dataset.companies_all)} Companies loaded')
            logger.info('====================================================')

        self.bG = dataset.bG_clean
        self.company_capability_graph = dataset.company_capability_graph
        self.capability_product_graph = dataset.capability_product_graph
        self.capability_graph = dataset.capability_graph
        self.company_country_graph = dataset.company_country_graph
        self.company_certification_graph = dataset.company_certification_graph
        self.G = dataset.G_clean
        self.companies_all = dataset.companies_all
        self.processes_all = list(dataset.processes_all)
        self.capabilities_all = list(dataset.capabilities_all)
        self.countries_all = dataset.countries_all
        self.certifications_all = dataset.certifications_all
        self.cG = dataset.cG_clean

        # Declare empty requirements for GNN libraries
        self.company_nodes = None
        self.process_nodes = None
        self.capability_nodes = None
        self.country_nodes = None
        self.certification_nodes = None
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

        # Same process for all Country Nodes
        self.country_nodes = \
            pd.DataFrame({'NODE_NAME': self.countries_all,
                          'NODE_TYPE': 'COUNTRY'})
        self.country_nodes['NODE_ID'] = \
            self.country_nodes.index.astype('int')

        # Same process for all Certification Nodes
        self.certification_nodes = \
            pd.DataFrame({'NODE_NAME': self.certifications_all,
                          'NODE_TYPE': 'CERTIFICATION'})
        self.certification_nodes['NODE_ID'] = \
            self.certification_nodes.index.astype('int')

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
                                     self.capability_nodes,
                                     self.country_nodes,
                                     self.certification_nodes
                                     ], ignore_index=True)

        all_nodes_frame.reset_index(drop=True, inplace=True)

        node_lookup_table = \
            all_nodes_frame[['NODE_NAME', 'NODE_ID']].set_index('NODE_NAME')
        node_lookup_table = node_lookup_table.reset_index()

        node_lookup_table['NODE_ID'] = \
            node_lookup_table['NODE_ID'].astype('int')

        del all_nodes_frame

        def sub_frame_generator(original_graph: nx.DiGraph,
                                relation_type: str,
                                src_type: str,
                                dst_type: str,
                                lookup_table: pd.DataFrame) -> pd.DataFrame:
            """
            Args:
                original_graph: Graph generated from dataset
                relation_type: Relationship type, e.g. 'buys_from'
                src_type: 'Company', 'Product', or 'Capability'
                dst_type: 'Company', 'Product', or 'Capability'
                lookup_table: lookup table parsed into function

            Returns:
                Dataframe -

              src | src_id| | dst| dst_id | relation_type | src_type | dst_type
            """
            sources_targets_nx = nx.to_pandas_edgelist(original_graph)

            empty_data_frame = pd.DataFrame({'src': [],
                                             'dst': [],
                                             'src_id': [],
                                             'dst_id': []})

            empty_data_frame['src'] = sources_targets_nx['source']
            empty_data_frame['dst'] = sources_targets_nx['target']

            mapper = dict(lookup_table[['NODE_NAME', 'NODE_ID']].values)
            empty_data_frame['src_id'] = empty_data_frame['src'].map(mapper)
            empty_data_frame['dst_id'] = empty_data_frame['dst'].map(mapper)

            # Let me know if any of the rows were not found in the mapper.
            if len(np.where(empty_data_frame.isna())[0]) > 0:
                raise Exception
            empty_data_frame = empty_data_frame.dropna()

            # Add in final pieces of information regarding relation type and src
            empty_data_frame['relation_type'] = relation_type
            empty_data_frame['src_type'] = src_type
            empty_data_frame['dst_type'] = dst_type
            return empty_data_frame

        ########################################################################
        # SUB-FRAME (Company -> buys_from -> Company)
        ########################################################################
        company_company_frame = \
            sub_frame_generator(self.G,
                                relation_type='buys_from',
                                src_type='company',
                                dst_type='company',
                                lookup_table=node_lookup_table)

        ########################################################################
        # SUB-FRAME (Company -> company_makes -> Product)
        ########################################################################
        company_product_frame = (
            sub_frame_generator(self.bG,
                                relation_type='company_makes',
                                src_type='company',
                                dst_type='product',
                                lookup_table=node_lookup_table)
        )

        ########################################################################
        # SUB-FRAME (Product -> complimentary_product_to -> Product)
        ########################################################################
        product_product_frame = (
            sub_frame_generator(self.cG,
                                relation_type='complimentary_product_to',
                                src_type='product',
                                dst_type='product',
                                lookup_table=node_lookup_table)
        )

        ########################################################################
        # SUB-FRAME (Capability -> capability_produces -> Product)
        ########################################################################
        capability_product_frame = (
            sub_frame_generator(self.capability_product_graph,
                                relation_type='capability_produces',
                                src_type='capability',
                                dst_type='product',
                                lookup_table=node_lookup_table)
        )

        ########################################################################
        # SUB-FRAME (Company -> has_capability -> Capability)
        ########################################################################
        company_capability_frame = (
            sub_frame_generator(self.company_capability_graph,
                                relation_type='has_capability',
                                src_type='company',
                                dst_type='capability',
                                lookup_table=node_lookup_table)
        )

        ########################################################################
        # SUB-FRAME (Company -> located_in -> Country)
        ########################################################################
        company_country_frame = (
            sub_frame_generator(self.company_country_graph,
                                relation_type='located_in',
                                src_type='company',
                                dst_type='country',
                                lookup_table=node_lookup_table)
        )

        ########################################################################
        # SUB-FRAME (Company -> has_cert -> Certification)
        ########################################################################
        company_certification_frame = (
            sub_frame_generator(self.company_certification_graph,
                                relation_type='has_cert',
                                src_type='company',
                                dst_type='certification',
                                lookup_table=node_lookup_table)
        )

        ########################################################################
        # Add everything together into a triplets frame
        ########################################################################
        self.triplets = pd.concat([company_company_frame,
                                   product_product_frame,
                                   capability_product_frame,
                                   company_product_frame,
                                   company_capability_frame,
                                   company_country_frame,
                                   company_certification_frame],
                                  ignore_index=True)

        logger.info('=========================================================')
        logger.info('\n')
        logging.info(f'{company_company_frame.head().to_string()}')
        logger.info('=========================================================')
        logger.info('\n')
        logging.info(f'{product_product_frame.head().to_string()}')
        logger.info('=========================================================')
        logger.info('\n')
        logging.info(f'{capability_product_frame.head().to_string()}')
        logger.info('=========================================================')
        logger.info('\n')
        logging.info(f'{company_product_frame.head().to_string()}')
        logger.info('=========================================================')
        logger.info('\n')
        logging.info(f'{company_capability_frame.head().to_string()}')
        logger.info('=========================================================')
        logger.info('\n')
        logging.info(f'{company_country_frame.head().to_string()}')
        logger.info('\n')
        logger.info('=========================================================')
        logging.info(f'{company_certification_frame.head().to_string()}')
        logger.info('====================================================')
        logger.info(f'The Knowledge Graph has {self.triplets.shape[0]} edges')

        # self.triplets.shape  #  = (796,124, 7)
        del company_company_frame, product_product_frame,\
            capability_product_frame, company_capability_frame, \
            company_country_frame, company_certification_frame

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
                                                'has_capability': 5,
                                                'located_in': 6,
                                                'has_cert': 7})
        )

        logger.info('Triplets created for all entities in the KG')
        # self.triplets.to_parquet('data/02_intermediate/triplets.parquet',
        #                          engine='pyarrow',
        #                          compression='gzip')
        logger.info('Saved triplets frame...')
        return self.triplets

    def process(self) -> None:
        """
        super method -> Gets called first.
        """
        if self.triplets_from_scratch:
            self.create_triples()
            logger.info('Triplets created. Starting processing to pytorch...')
        else:
            self.triplets = \
                pd.read_parquet(self.data_path + 'triplets.parquet')
            logger.info('Triplets loaded from memory, processing to torch...')
            self.load()

        ########################################################################
        # Create Heterograph from DGL - This process is done iteratively
        ########################################################################
        if self.load_graph:
            self.load()
        else:
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

            cond = self.triplets['relation_type'] == 'located_in'
            company_location = self.triplets.loc[cond]
            company_location_triples = \
                [torch.tensor((int(src_id), int(dst_id)),
                              dtype=torch.int32) for src_id, dst_id
                 in zip(company_location.src_id, company_location.dst_id)]

            cond = self.triplets['relation_type'] == 'has_cert'
            company_cert = self.triplets.loc[cond]
            company_cert_triples = \
                [torch.tensor((int(src_id), int(dst_id)),
                              dtype=torch.int32) for src_id, dst_id
                 in zip(company_cert.src_id, company_cert.dst_id)]

            del makes_product, product_product, buys_from, cond
            data_dict = {
                ('company', 'buys_from', 'company'): company_buying_triples,
                ('company', 'makes_product', 'product'): makes_product_triples,
                ('product', 'complimentary_product_to', 'product'): product_product_triples,
                ('capability', 'capability_produces', 'product'): capability_product_triples,
                ('company', 'has_capability', 'capability'): company_capability_triples,
                ('company', 'located_in', 'country'): company_location_triples,
                ('company', 'has_cert', 'certification'): company_cert_triples
            }

            self.graph = dgl.heterograph(data_dict)
            self.num_rels = len(self.graph.etypes)
            self.save()

    def save(self):
        # save graphs and labels
        graph_path = os.path.join(self.data_path, '_dgl_graph.bin')
        dgl.save_graphs(graph_path, self.graph)
        # save other information in python dict
        info_path = os.path.join(self.data_path, '_info.pkl')
        dgl.data.utils.save_info(info_path, {'num_rels': self.num_rels})

    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.data_path, '_dgl_graph.bin')
        self.graph, label_dict = dgl.data.utils.load_graphs(graph_path)
        info_path = os.path.join(self.data_path, '_info.pkl')
        self.num_rels = dgl.data.utils.load_info(info_path)['num_rels']

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
