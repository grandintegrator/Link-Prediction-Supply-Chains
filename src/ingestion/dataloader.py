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
        if params.load_graph:
            self.full_graph = loader[0][0]
        else:
            self.full_graph = loader[0]
        self.edge_types = self.full_graph.etypes

        self.nodes = ['company',
                      'product',
                      'certification',
                      'country',
                      'capability']

        # Class initialisations
        self.training_data = None
        self.testing_data = None
        self.valid_data = None

    def get_training_testing(self) -> None:
        """
        # Splits the edges into an independent training, validation, and testing
        set for inference.
        """
        # randomly generate training masks for our buys_from edges
        # Need to make sure this is reproducible.

        edge_types = self.full_graph.etypes
        graph_schema = self.full_graph.canonical_etypes
        train_edge_split_dict = {}
        valid_edge_split_dict = {}
        test_edge_split_dict = {}

        for edge_type, edge_tuple in zip(edge_types, graph_schema):
            src, dst = self.full_graph.edges(etype=edge_type)
            # Cheeky assertion to ensure that source and destination edges are
            # maintained.
            assert len(src) == len(dst)

            test_size = int(self.params.test_p * len(src))
            valid_size = int(self.params.valid_p*len(src))
            train_size = len(src) - valid_size - test_size

            # Source and destinations for training
            src_train = src[0:train_size]
            dst_train = dst[0:train_size]
            # Source and destinations for validation
            src_valid = src[train_size: valid_size + train_size]
            dst_valid = dst[train_size: valid_size + train_size]
            # Source and destinations for testing
            src_test = src[valid_size + train_size:]
            dst_test = dst[valid_size + train_size:]

            train_edge_split_dict[edge_tuple] = (src_train, dst_train)
            valid_edge_split_dict[edge_tuple] = (src_valid, dst_valid)
            test_edge_split_dict[edge_tuple] = (src_test, dst_test)

        self.training_data = dgl.heterograph(train_edge_split_dict)
        self.valid_data = dgl.heterograph(valid_edge_split_dict)
        self.testing_data = dgl.heterograph(test_edge_split_dict)

    def get_training_dataloader(self) -> dgl.dataloading.EdgeDataLoader:
        # Create the sampler object
        self.get_training_testing()

        for node in self.nodes:
            n_node_type = self.training_data.num_nodes(node)
            self.training_data.nodes[node].data['feature'] = (
                torch.randn(n_node_type,
                            self.params.num_node_features)
            )

        graph_eid_dict = \
            {etype: self.training_data.edges(etype=etype, form='eid')
             for etype in self.training_data.etypes}

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        negative_sampler = dgl.dataloading.negative_sampler.Uniform(10)

        train_data_loader = dgl.dataloading.EdgeDataLoader(
            self.training_data, graph_eid_dict, sampler,
            negative_sampler=negative_sampler,
            batch_size=self.params.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=self.params.num_workers)
        return train_data_loader

    def get_test_data_loader(self) -> dgl.dataloading.EdgeDataLoader:
        # Creates testing data loader for evaluation
        self.get_training_testing()

        if self.params.eval_type == 'validation':
            for node in self.nodes:
                n_node_type = self.valid_data.num_nodes(node)
                self.valid_data.nodes[node].data['feature'] = (
                    torch.randn(n_node_type,
                                self.params.num_node_features)
                )

            graph_eid_dict = \
                {etype: self.valid_data.edges(etype=etype, form='eid')
                 for etype in self.valid_data.etypes}

            # sampler = dgl.dataloading.MultiLayerFullNeighborSampler([30, 30])
            # sampler = (
            #     dgl.dataloading.MultiLayerNeighborSampler([60, 60])
            # )
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
            negative_sampler = dgl.dataloading.negative_sampler.Uniform(10)

            valid_data_loader = dgl.dataloading.EdgeDataLoader(
                self.valid_data, graph_eid_dict, sampler,
                negative_sampler=negative_sampler,
                batch_size=self.params.batch_size,
                shuffle=True,
                drop_last=False,
                pin_memory=True,
                num_workers=self.params.num_workers)
            return valid_data_loader
        else:
            for node in self.nodes:
                n_node_type = self.testing_data.num_nodes(node)
                self.testing_data.nodes[node].data['feature'] = (
                    torch.randn(n_node_type,
                                self.params.num_node_features)
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
                batch_size=self.params.batch_size,
                shuffle=True,
                drop_last=False,
                # pin_memory=True,
                num_workers=self.params.num_workers)
            return test_data_loader
