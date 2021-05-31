import dgl
import torch

from ingestion.dgl_dataset import SupplyKnowledgeGraphDataset


class SCDataLoader(object):
    def __init__(self, params):
        self.params = params
        loader = SupplyKnowledgeGraphDataset(params=params,
                                             path='data/02_intermediate/',
                                             from_scratch=False,
                                             triplets_from_scratch=False,
                                             load_graph=True)
        self.full_graph = loader[0][0]
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
        # edge_type_1 = ('company', 'buys_from', 'company')
        # edge_type_2 = ('company', 'makes_product', 'product')
        # edge_type_3 = ('product', 'product_product', 'product')
        # edge_type_3 = ()

        edge_type_1 = ('company', 'buys_from', 'company')
        edge_type_2 = ('company', 'makes_product', 'product')
        edge_type_3 = ('product', 'complimentary_product_to', 'product')
        edge_type_4 = ('capability', 'capability_produces', 'product')
        edge_type_5 = ('company', 'has_capability', 'capability')
        edge_type_6 = ('company', 'located_in', 'country')
        edge_type_7 = ('company', 'has_cert', 'certification')

        train_data_dict = {
            edge_type_1: (src_train, dst_train),
            edge_type_2: self.full_graph.edges(etype='makes_product'),
            edge_type_3: self.full_graph.edges(etype='complimentary_product_to'),
            edge_type_4: self.full_graph.edges(etype='capability_produces'),
            edge_type_5: self.full_graph.edges(etype='capability_produces'),
            edge_type_6: self.full_graph.edges(etype='located_in'),
            edge_type_7: self.full_graph.edges(etype='has_cert')
        }

        test_data_dict = {
            edge_type_1: (src_test, dst_test),
            edge_type_2: self.full_graph.edges(etype='makes_product'),
            edge_type_3: self.full_graph.edges(etype='complimentary_product_to'),
            edge_type_4: self.full_graph.edges(etype='capability_produces'),
            edge_type_5: self.full_graph.edges(etype='capability_produces'),
            edge_type_6: self.full_graph.edges(etype='located_in'),
            edge_type_7: self.full_graph.edges(etype='has_cert')
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
