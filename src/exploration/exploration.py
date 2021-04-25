# visualize graphs, sample
import os
import pandas as pd
import numpy as np
import pickle

from Marklines import Marklines
from typing import List, Dict, Any
from jgraph import *
from collections import Counter
from random import sample
from tqdm import tqdm
from networkx.algorithms import bipartite
from statistics import median, mean

import logging
import itertools

import plotly.express as px
import plotly.io as pio
import plotly.figure_factory as ff
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

# Plotting preferences
pio.templates.default = "plotly_white"

# Pandas debugging options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Logger preferences
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig()


class VisualiseGraph(object):
    def __init__(self, path: str = 'data/02_intermediate/marklinesEdges.p'):
        """
        Args:
            path: Location of the MarkLines pickled object
        """

        # load existing graph object
        self.multi_pair_frame = None
        self.graph_object = pickle.load(open(path, "rb"))
        logger.info('NetworkX graph loaded')

    # import networkx as nx
    # nx.is_bipartite()
    # @staticmethod
    # def get_summary_statistics(self) -> Dict[str, Any]:
    #     """Returns some traditional summary statistics at the graph level
    #     """
    #     logger.info('Generating graph statistics...')

    def get_degree_distribution(self,
                                out_path: str = 'data/04_results/',
                                loglog: bool = True) -> None:
        """Generates the degree distribution
        Args:
            out_path: .html file location
            loglog: Plot log-log scatter plot showing power law distribution?

        """
        degrees = (
            [self.graph_object.G.degree(n) for n in self.graph_object.G.nodes()]
        )
        fig = ff.create_distplot([degrees], group_labels=['Degree Distribution'])
        fig.update_layout(font_family='Arial',
                          title='Degree Distribution of Automotive Network',
                          yaxis_title=r"P(n)",
                          xaxis_title=r"n - Node Degree",
                          # legend_title='Legend',
                          font=dict(size=16))
        fig.write_html(out_path + 'G_degree_distribution.html')

        if loglog:
            # User collections.Counter to get a count of the degrees
            counts = dict(Counter(degrees))
            loglog_df = \
                pd.DataFrame.from_dict(counts, orient='index').reset_index()
            loglog_df = loglog_df.rename(columns={'index': 'DEGREE_COUNT',
                                                  0: 'FRACTION_OF_NODES'})

            number_of_nodes = len(list(self.graph_object.G.nodes))
            loglog_df['FRACTION_OF_NODES'] = loglog_df['FRACTION_OF_NODES']/number_of_nodes
            fig = px.scatter(data_frame=loglog_df,
                             x='DEGREE_COUNT',
                             y='FRACTION_OF_NODES',
                             log_x=True,
                             log_y=True)
            fig.update_layout(font_family='Arial',
                              title='Log Log of Degree Distribution',
                              yaxis_title='Fraction of Nodes (log)',
                              xaxis_title='Degree (log)'
                              )
            fig.write_html(out_path + 'G_log_log.html')

    def build_igraph_plot(self, out_path: str = 'data/04_results/') -> None:
        """Function converts the NetworkX network to iGraph and then saves a
        visualisation
        Args:
            out_path: Storage location of output eps file

        """
        igraph_object = Graph()
        converted_igraph = igraph_object.from_networkx(g=self.graph_object.G)

        self.graph_object.rawdf['CompanyName'].isin(self.graph_object.G.nodes)

        visual_style = dict()
        visual_style["edge_curved"] = False

        # Set the layout
        my_layout = converted_igraph.layout_auto()
        visual_style["layout"] = my_layout

        # Plot and save the graph
        plot(converted_igraph, out_path+'graph.eps', **visual_style)

    def create_multi_pair_frame(self,
                                sample_portion: float = 3,
                                product_product_n: int = 10) -> pd.DataFrame:
        """Function uses the bG and G graphs within the graph_object
        to create a multi relational dataframe

        subject | object | relation | subject_type | object_type

        Args:
            sample_portion: percentage of overall graph to sample for analysis
            product_product_n: product_product graph subsample size for viz
        """
        edges_to_sample = sample_portion  # round(sample_portion*len(self.graph_object.G.edges))

        # Sample a set of edges from the buy-sell graph
        sampled_edges = sample(self.graph_object.G.edges, k=edges_to_sample)

        ########################################################################
        # Create buy-sell company-company sub-frame
        ########################################################################
        subjects_companies, objects_companies = map(list, zip(*sampled_edges))

        companies_relations = pd.DataFrame({'subjects': subjects_companies,
                                            'objects': objects_companies,
                                            'relation_type': 'buys_from',
                                            'subject_type': 'Company',
                                            'object_type': 'Company'})

        ########################################################################
        # Create product_company sub-frame
        ########################################################################
        all_companies_sample = subjects_companies + objects_companies

        process_relations = (
            [el for el in self.graph_object.bG.edges if el[1] in all_companies_sample]
        )

        subjects_process, objects_companies = map(list, zip(*process_relations))

        products_relations = pd.DataFrame({'subjects': subjects_process,
                                           'objects': objects_companies,
                                           'relation_type': 'makes_product',
                                           'subject_type': 'Process',
                                           'object_type': 'Company'})

        ########################################################################
        # Create product-product sub frame - like protein-protein network
        ########################################################################

        process_process_relations = (
            [el for el in self.graph_object.cG.edges
             if el[0] in subjects_process or el[1] in subjects_process]
        )

        subject_process, object_process = \
            map(list, zip(*process_process_relations))

        process_process_relations = pd.DataFrame({'subjects': subject_process,
                                                  'objects': object_process,
                                                  'relation_type': 'product_product',
                                                  'subject_type': 'Process',
                                                  'object_type': 'Process'})

        process_process_relations = \
            process_process_relations.sample(n=product_product_n)

        self.multi_pair_frame = pd.concat([companies_relations,
                                           products_relations,
                                           process_process_relations])

        return self.multi_pair_frame

    @staticmethod
    def plot_distribution(dist,
                          title: str = "",
                          x_label: str = "",
                          y_label: str = "",
                          file_name: str = None,
                          bins: int = 10):
        plt.figure(figsize=(6, 3.5))
        sns.set_context("paper", font_scale=1.8)
        sns.set_style('ticks')
        sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
        sns.histplot(dist, kde=False, color=sns.xkcd_rgb['red'],
                     bins=bins, alpha=1, log_scale=True)
        plt.xlabel(x_label)
        plt.title(title)
        plt.tight_layout()
        plt.gcf().subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
        plt.ylabel(y_label)
        plt.show()
        if file_name:
            plt.savefig(file_name)

    def get_products(self, company: str) -> List:
        """
        Args:
            company: takes a string input and returns the products the company
                creates

        Returns:
            List of products
        """
        products_from_company = (
            [el for el in self.graph_object.bG.edges if el[1] == company]
        )
        try:
            products_company, _ = map(list, zip(*products_from_company))
        except ValueError:
            products_company = []

        return products_company

    def product_overlap_chart(self) -> None:
        """Function loops through links and non-links to find the percentage
        of overlap between shared products if a link exists
        Returns:
            None
        """
        ########################################################################
        # Analysis from known edges within the graph - bipartite projection
        ########################################################################
        # Get the unique company nodes and product nodes  from bG graph
        company_nodes = list(set([x[1] for x in self.graph_object.bG.edges]))
        product_nodes = list(set([x[0] for x in self.graph_object.bG.edges]))

        company_nodes = list(set(company_nodes) - set(product_nodes))

        company_nodes = company_nodes[1:10]
        # Weighted bipartite projection - weights contain shared edges
        # TODO: smaller subset on BOTH sides
        # TODO:
        bipartite_projection = (
            bipartite.weighted_projected_graph(self.graph_object.bG,
                                               nodes=company_nodes)
        )
        nx.is_bipartite(self.graph_object.bG)
        # # Get the weight values from the bipartite projection
        # weights_with_edges = \
        #     list(nx.get_edge_attributes(bipartite_projection, "weight").values())
        #
        # weight_frame = (
        #     pd.DataFrame({'weights_with_edges': weights_with_edges[::100],
        #                   'weights_without_edges': weights_with_edges[::100]})
        # )

        # adjacency_matrix = nx.adj_matrix(bipartite_projection)
        # edge_list_bipartite = \
        #     nx.convert_matrix.to_pandas_edgelist(bipartite_projection)

        # Get the Adjacency matrix of the bipartite projection graph
        adjacency_bipartite = (
            nx.to_pandas_adjacency(bipartite_projection)
        )

        del bipartite_projection
        ########################################################################
        # Non-existing links --- using the adjacency matrix 0s entries
        ########################################################################
        # Get only the companies who don't share a common product
        no_shared_products_adjacency = (
            adjacency_bipartite[adjacency_bipartite == 0]
        )
        # Get candidates for checking
        no_shared_products_list = [(i, j) for i, j in zip(no_shared_products_adjacency.index,
                                                          no_shared_products_adjacency.columns)]
        # no_shared_products_list = list((no_shared_products_adjacency.index,
        #                                 no_shared_products_adjacency.columns))

        candidate_edges_no_shared_products = list(set(no_shared_products_list))
        print(candidate_edges_no_shared_products)
        # Flush memory

        ########################################################################
        # Summarise for when companies do buy-sell from one another
        ########################################################################
        # nx.to_pandas_edgelist()
        # edge_list_bipartite_single == no common edges amongst node-source
        # edge_list_bipartite_single = (
        #     adjacency_bipartite.loc[adjacency_bipartite['weight'] == 0, :]
        # )
        # # edge_list_bipartite_rest == >1 common edges amongst node-source
        # edge_list_bipartite_rest = (
        #     adjacency_bipartite.loc[~(adjacency_bipartite['weight'] == 0), :]
        # )
        #
        # # Loop through source, target in no links projection
        # for source, target in zip(edge_list_bipartite_single.head()['source'],
        #                           edge_list_bipartite_single.head()['target']):
        #     if (source, target) in self.graph_object.G.edges:
        #         print('single-edge')
        #


        # non_edges = []
        # num = 0
        # for _ in tqdm(range(25000)):
        #     random_selected_company = np.random.choice(product_nodes, size=1,
        #                                                replace=False)
        #     random_selected_product = np.random.choice(company_nodes, size=1,
        #                                                replace=False)
        #     pair_nodes = (random_selected_company[0], random_selected_product[0])
        #     if pair_nodes not in self.graph_object.bG.edges:
        #         num += 1
        #         non_edges.append(list(pair_nodes))
        #
        # # Create a bipartite projection of the anti graph
        # bg_anti = nx.from_edgelist(non_edges)
        # company_nodes_anti = [x[1] for x in list(bg_anti.edges)]
        # anti_bipartite_projection = (
        #     bipartite.weighted_projected_graph(bg_anti,
        #                                        nodes=company_nodes_anti)
        # )
        # weights_no_edges = \
        #     nx.get_edge_attributes(anti_bipartite_projection, "weight").values()
        # new_weight_frame = pd.DataFrame()
        # if len(weights_no_edges) < weight_frame['weights_with_edges'].shape[0]:
        #     new_weight_frame['weights_with_edges'] = np.random.choice(
        #         weight_frame['weights_with_edges'], size=len(weights_no_edges),
        #         replace=False
        #     )
        #     new_weight_frame['weights_without_edges'] = weights_no_edges
        # else:
        #     new_weight_frame['weights_with_edges'] = \
        #         weight_frame['weights_with_edges']
        #
        #     new_weight_frame['weights_without_edges'] = np.random.choice(
        #         list(weights_no_edges),
        #         size=new_weight_frame['weights_with_edges'].shape[0],
        #         replace=False
        #     )
        #
        # del weight_frame, anti_bipartite_projection
        ########################################################################
        # Plot the difference in the distributions for contrasting
        ########################################################################
        # self.plot_distribution(new_weight_frame['weights_with_edges'],
        #                        title='Number of shared product neighbours',
        #                        x_label='Number of Shared Products',
        #                        y_label='Count',
        #                        bins=15)
        #
        # sns.histplot(data=new_weight_frame,
        #              x='weights_without_edges',
        #              log_scale=True,
        #              bins=15)
        # plt.show()
        #
        # sns.histplot(
        #     new_weight_frame.melt(),
        #     x="value", hue="variable",
        #     multiple="stack",
        #     palette="light:m_r",
        #     edgecolor=".3",
        #     linewidth=.5,
        #     log_scale=True,
        # )
        # plt.show()




if __name__ == '__main__':
    graph_class = VisualiseGraph()
    graph_class.product_overlap_chart()
    # graph_class.get_degree_distribution()
    # graph_class.create_multi_pair_frame(sample_portion=2,
    #                                     product_product_n=10)

# integrated gradients
