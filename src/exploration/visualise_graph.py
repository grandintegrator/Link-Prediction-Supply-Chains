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
        from Marklines import Marklines
        # load existing graph object
        self.multi_pair_frame = None
        self.graph_object = pickle.load(open(path, "rb"))
        logger.info('NetworkX graph loaded')

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
                          font=dict(size=24))
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
                              xaxis_title='Degree (log)',
                              font=dict(size=24))

            fig.write_html(out_path + 'G_log_log.html')

    def build_igraph_plot(self, out_path: str = 'data/04_results/') -> None:
        """Function converts the NetworkX network to iGraph and then saves a
        visualisation
        Args:
            out_path: Storage location of output eps file

        """
        from igraph import Graph
        from igraph import plot

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
        import dask.dataframe as dd
        ########################################################################
        # Analysis from known edges within the graph - bipartite projection
        ########################################################################
        # Get the unique company nodes and product nodes  from bG graph
        # company_nodes = list(set([x[1] for x in self.graph_object.bG.edges]))
        # product_nodes = list(set([x[0] for x in self.graph_object.bG.edges]))
        #
        # company_nodes = list(set(company_nodes) - set(product_nodes))

        # Weighted bipartite projection - weights contain shared edges
        companies = set(self.graph_object.supplierProductdf['companyName'].values)
        products = set(self.graph_object.supplierProductdf['product'].values)

        # Intersected products that should never have been there
        intersection_products = products.intersection(companies)
        del companies, products
        # Converted to parquet before the below line - engine=pyarrow, gzip
        supplier_product_df_fixed = (
            self.graph_object.supplierProductdf
                .loc[~self.graph_object.supplierProductdf['companyName']
                .isin(intersection_products)]
        )

        supplier_product_df_fixed = supplier_product_df_fixed.loc[1:10000, :]

        # About 2k rows were duplicated, not much, but something!
        supplier_product_df_fixed = supplier_product_df_fixed.drop_duplicates()

        # Create an empty NetworkX graph
        bipartite_graph = nx.Graph()

        # Add in data into empty graph object
        for company, product in zip(supplier_product_df_fixed['companyName'],
                                    supplier_product_df_fixed['product']):
            # Add edges one by one into a new graph which will be bipartite
            bipartite_graph.add_edge(company, product)

        assert nx.is_bipartite(bipartite_graph)

        projected_nodes = supplier_product_df_fixed['companyName'].unique()
        bipartite_projection = (
            bipartite.weighted_projected_graph(bipartite_graph,
                                               nodes=projected_nodes)
        )
        del projected_nodes
        logger.info('Create weighted projected graph.')

        # Get the Adjacency matrix of the bipartite projection graph
        adjacency_bipartite = (
            nx.to_pandas_adjacency(bipartite_projection)
        )
        logger.info('Created adjacency matrix..')
        # Convert to a dask dataframe.
        adjacency_bipartite = dd.from_pandas(adjacency_bipartite,
                                             npartitions=100)

        del bipartite_projection, bipartite_graph, supplier_product_df_fixed
        logger.info('Cleaned cache and starting compute heavy work')
        ########################################################################
        # Non-existing links --- using the adjacency matrix 0s entries
        ########################################################################
        no_shared_products_candidates = []
        shared_products_with_connection = 0
        for col in tqdm(adjacency_bipartite.columns):
            # Handle the no shared weights in projection side first:
            zero_cond = adjacency_bipartite[col] == 0
            zero_connections = adjacency_bipartite.loc[zero_cond].index
            # zero_connections = adjacency_bipartite.index[zero_cond]
            candidates = [(col, new_product) for new_product in zero_connections.compute()
                          if col != new_product]
            for candidate in candidates:
                if self.graph_object.G.has_edge(candidate[0], candidate[1]):
                    # Number of edges out of all edges that had no sharing
                    shared_products_with_connection += 1

            # shared_weights = adjacency_bipartite.index[~zero_cond]
            shared_weights = adjacency_bipartite.loc[~zero_cond].index
            candidates = [(col, new_product) for new_product in shared_weights.compute()
                          if col != new_product]
            for candidate in candidates:
                if self.graph_object.G.has_edge(candidate[0], candidate[1]):
                    shared_weight = adjacency_bipartite.loc[candidate[0],
                                                            candidate[1]]
                    no_shared_products_candidates.extend([shared_weight])

        logger.info(f'Found {shared_products_with_connection:.2f} shares')
        percentage_shared = (
            shared_products_with_connection/len(self.graph_object.G.edges())*100
        )
        logger.info(f"Implying only {percentage_shared:.2f}% of links didn't"
                    f" share products")
        with open('data/02_intermediate/no_shared_products_candidates.pkl',
                  'wb') as f:
            pickle.dump(no_shared_products_candidates, f)

        del adjacency_bipartite

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
    logger = logging.getLogger(__name__)
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    graph_class = VisualiseGraph()
    graph_class.product_overlap_chart()
    # graph_class.get_degree_distribution()
    # graph_class.create_multi_pair_frame(sample_portion=2,
    #                                     product_product_n=10)

