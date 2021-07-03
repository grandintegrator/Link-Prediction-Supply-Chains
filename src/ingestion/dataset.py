import networkx as nx
import pickle
import tqdm
import pandas as pd
import logging
import plotly.express as px
import plotly.io as pio
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt

from copy import copy
from networkx.algorithms import bipartite
from itertools import product, chain
from collections import Counter
# from utils import cleanCompany

# Plotting preferences
pio.templates.default = "plotly_white"

# Logger preferences
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig()
logger = logging.getLogger(__name__)


class KnowledgeGraphGenerator(object):
    def __init__(self, params, path: str = '../data/02_intermediate/'):
        """"
        This class looks at the existing graphs, G, bG, and cG
        couples that information with what we know about capabilities to create

        Returns
        -------
        G: (company -> buys_from -> company)
        bG: (company -> makes_product -> product)
        company_capability_graph: (company -> has_capability -> capability)
        capability_product_graph: (capability -> makes_product -> product)
        location based graph: (company -> located_in -> country)

        Graphs based on Bipartite projection (TBD):
        cG (product -> product) ---- Choosing to remove
        capability_graph (capability -> capability) ------- TBD..

        """
        self.params = params
        self.save_graph_path = self.params.graph_save_path
        ########################################################################
        # Load pickled objects from Edward's analysis
        ########################################################################
        with open(path + 'G.pickle', 'rb') as f:
            self.G = pickle.load(f)

        with open(path + 'cG.pickle', 'rb') as f:
            self.cG = pickle.load(f)

        if path == '../data/02_intermediate/':
            self.supplier_product_ds = \
                pd.read_parquet('../data/01_raw/supplier_product_df.parquet')
        else:
            self.supplier_product_ds = \
                pd.read_parquet('data/01_raw/supplier_product_df.parquet')
        # self.raw_df = \
        #     pd.read_pickle('../data/02_intermediate/raw_df_clean_co.pkl')
        self.raw_df = \
            pd.read_parquet(path + 'raw_df_truncated.parquet')

        ########################################################################
        # Create empty graph objects to store data into
        ########################################################################
        self.bG = nx.DiGraph()
        self.bG_clean = nx.DiGraph()
        self.G_clean = nx.DiGraph()
        self.cG_clean = nx.DiGraph()

        self.company_capability_graph = nx.DiGraph()
        self.capability_product_graph = nx.DiGraph()
        self.capability_graph = nx.DiGraph()
        self.company_country_graph = nx.DiGraph()
        self.company_certification_graph = nx.DiGraph()

        self.processes_all = None
        self.companies_all = None
        self.countries_all = None
        self.certifications_all = None

        self.capabilities_all = \
            ["Stamping", "Assembly", "Machining",
             "Plastic injection molding", "Welding",
             "Cold forging", "Plastic molding", "Heat treatment",
             "Iron casting", "Casting", "Hot forging",
             "Aluminum casting", "Aluminum die casting",
             "Plating", "Foaming", "Paint Coating",
             "Aluminum machining", "Die casting",
             "Various surface treatment", "Blow molding",
             "Rubber injection molding parts",
             "Electronics/Electric Parts", "Rubber parts",
             "Rubber-metal parts", "Rubber extruded parts",
             "Interior trim parts", "Exterior parts",
             "Forging", "Interior Trim Parts",
             "Plastic extruded parts", "Roll forming",
             "General Commodity", "Fine blanking",
             "Surface treatment/Heat treatment",
             "Various Processes", "Hydroforming"]

        self.capabilities_all = [el.title() for el in self.capabilities_all]

    def cut_cg(self):
        """Function cuts and removes weights from the cG graph that are too high
        Returns: cG for further analysis but without weights that are too high

        """
        cut_threshold = self.params.cg_weight_cut

        cg_graph = \
            nx.to_pandas_edgelist(self.cG)

        # Define the condition for cutting
        cond = cg_graph['weight'] >= cut_threshold
        cg_graph = cg_graph.loc[cond]

        # Reset the capability_product_graph
        self.cG = nx.DiGraph()
        edge_bunch = cg_graph[['source', 'target']].values
        self.cG.add_edges_from(ebunch_to_add=edge_bunch)

    def create_bg_clean(self):
        """
        Fix bG overlap issue (companies -> products)
        Companies -> Products should be bipartite but has products in companies
        and companies in products.
        1) Remove the overlap between the two bipartite sides.
        2) Remove capabilities from the product side.

        Returns
        -------
        Cleaned bG - bipartite with all capabilities removed from product side
        """
        logger.info('Creating bG_clean graph and saving all capabilities.')
        self.supplier_product_ds = \
            self.supplier_product_ds.apply(lambda x: x.str.title())

        # Create original dirty bG
        self.bG.add_edges_from([(u, v)
                                for u, v in self.supplier_product_ds.values])

        # Now clean the original bG and create bG_clean
        suppliers_bg = set(self.supplier_product_ds['companyName'].values)
        products_bg = set(self.supplier_product_ds['product'].values)

        products_in_source = list(suppliers_bg.intersection(products_bg))
        # Remove the products that are in the companies column
        self.supplier_product_ds = (
            self.supplier_product_ds
                .loc[~self.supplier_product_ds['companyName']
                .isin(products_in_source)]
        )
        # self.supplier_product_ds.shape Out[63]: (205400, 2)
        # Now remove any rows that contain a capability - captured elsewhere
        cond = (
            self.supplier_product_ds['companyName'].isin(self.capabilities_all)
            | self.supplier_product_ds['product'].isin(self.capabilities_all)
        )
        indices_drop = self.supplier_product_ds[cond].index

        # Delete these row indexes from dataframe
        self.supplier_product_ds = \
            self.supplier_product_ds.drop(indices_drop, inplace=False)

        # self.supplier_product_ds.shape
        # Out[15]: (120719, 2)

        self.bG_clean.add_edges_from(self.supplier_product_ds.values)

    def generate_new_graphs(self) -> None:
        """
        Creates capability based graphs.
        """
        self.create_bg_clean()
        ########################################################################
        # Find capability nodes from cG (product-product) edges and bG edge
        # WE KNOW THE SET BELOW WILL CONTAIN PRODUCTS AND CAPABILITIES.
        # THE CODE BELOW JUST FINDS ANYTHING THAT IS NOT A COMPANY.
        ########################################################################
        # Collect all possible processes in the datasets from cG and bG edges
        process_nodes_names_subjects = set(([el[0] for el in self.cG.edges]))
        process_nodes_names_objects = set(([el[1] for el in self.cG.edges]))
        process_nodes_names_subjects_bg = set(([el[1] for el in self.bG.edges]))

        # Add all of the process - related nodes and remove capabilities
        process_nodes_set = list(
            process_nodes_names_subjects |
            process_nodes_names_objects |
            process_nodes_names_subjects_bg
        )
        # Convert to title case and find the intersection in sets
        process_nodes_set = set([el.title() for el in process_nodes_set])
        capability_set = set([el.title() for el in self.capabilities_all])

        capabilities_found = \
            list(process_nodes_set.intersection(capability_set))

        ########################################################################
        # Create Capability graph - (Capability -> Product)
        ########################################################################
        # for edge in tqdm.tqdm(self.cG.edges):
        #     p1 = edge[0].title()
        #     p2 = edge[1].title()
        #
        #     if (p1 in capabilities_found) and (p2 not in capabilities_found):
        #         self.capability_product_graph.add_edge(u_of_edge=p1,
        #                                                v_of_edge=p2)
        #     elif (p2 in capabilities_found) and (p1 not in capabilities_found):
        #         self.capability_product_graph.add_edge(u_of_edge=p2,
        #                                                v_of_edge=p1)

        ########################################################################
        # Create Company-Capability graph - (Company -> Capability)
        ########################################################################
        for edge in tqdm.tqdm(self.bG.edges):
            p1 = edge[0].title()
            p2 = edge[1].title()

            if p1 in capabilities_found:
                self.company_capability_graph.add_edge(u_of_edge=p2,
                                                       v_of_edge=p1)
            elif p2 in capabilities_found:
                self.company_capability_graph.add_edge(u_of_edge=p1,
                                                       v_of_edge=p2)

    def clean_and_generate_graphs(self) -> None:
        """
        Function looks at the newly created (clean) graphs to clean
            G, bG, and cG before converting into a DGL dataset.
        """
        self.generate_new_graphs()

        ########################################################################
        # Get rid of Capabilities or products from the G graph
        # This graph should only contain companies - NOTHING ELSE.
        # THIS PROCESS TOOK self.G: 89,013 ---> 88,997 edges (16 edges)
        ########################################################################
        self.processes_all = [el[1] for el in self.bG_clean.edges]

        g_edge_df = nx.to_pandas_edgelist(self.G)
        g_edge_df = g_edge_df.apply(lambda x: x.str.title())

        cond_src = (
            g_edge_df['source'].isin(self.capabilities_all)
            | g_edge_df['source'].isin(self.processes_all)
        )
        cond_dst = (
                g_edge_df['target'].isin(self.capabilities_all)
                | g_edge_df['target'].isin(self.processes_all)
        )
        g_edge_df = g_edge_df.loc[~(cond_src | cond_dst)]

        self.G_clean.add_edges_from([(u, v) for u, v in g_edge_df.values])

        self.companies_all = list(
            set([el[1].title() for el in self.G_clean.edges]) |
            set([el[0].title() for el in self.G_clean.edges]) |
            set([el[0] for el in self.bG_clean.edges]) |
            set([el[0] for el in self.company_capability_graph.edges])
            # set([el[0]])
        )

        ########################################################################
        # Fix cG by getting rid of all product -> products as capabilities
        # len(self.cG.edges)
        #           Goes from 472,978 ---> 353,060 (25% reduction)
        ########################################################################
        cg_edge_df = nx.to_pandas_edgelist(self.cG)
        cg_edge_df = \
            cg_edge_df.apply(lambda x: x.str.title())
        cond = (
            cg_edge_df['source'].isin(self.capabilities_all)
            | cg_edge_df['source'].isin(self.companies_all)
            | cg_edge_df['target'].isin(self.capabilities_all)
            | cg_edge_df['target'].isin(self.companies_all)
        )
        cg_edge_df = cg_edge_df.loc[~cond]
        self.cG_clean.add_edges_from([(u, v) for u, v in cg_edge_df.values])

        ########################################################################
        # Clean Company-Capability graph - (Company -> Capability)
        # 83,793 ---> 50,676
        ########################################################################
        company_capability_edge_df = (
            nx.to_pandas_edgelist(self.company_capability_graph)
        )
        cond_drop = (
            company_capability_edge_df['source'].isin(self.processes_all)
            | company_capability_edge_df['source'].isin(self.capabilities_all)
            | company_capability_edge_df['target'].isin(self.companies_all)
            | company_capability_edge_df['target'].isin(self.processes_all)
        )
        company_capability_edge_df = company_capability_edge_df.loc[~cond_drop]
        self.company_capability_graph = nx.DiGraph()
        edge_bunch = [(u, v) for u, v in company_capability_edge_df.values]
        self.company_capability_graph.add_edges_from(edge_bunch)

        ########################################################################
        # CLEAN Capability graph - (Capability -> Product)
        # 51348 --> 16326
        ########################################################################
        capability_product_graph_edge_df = (
            nx.to_pandas_edgelist(self.capability_product_graph)
        )
        cond_drop = (
            capability_product_graph_edge_df['source'].isin(self.processes_all)
            | capability_product_graph_edge_df['source'].isin(self.companies_all)
            | capability_product_graph_edge_df['target'].isin(self.companies_all)
            | capability_product_graph_edge_df['target'].isin(self.capabilities_all)
        )
        capability_product_graph_edge_df = (
            capability_product_graph_edge_df.loc[~cond_drop]
        )
        edges_add = capability_product_graph_edge_df.values
        self.capability_product_graph = nx.DiGraph()
        self.capability_product_graph\
            .add_edges_from([(u, v) for u, v in edges_add])

    def analyse_bipartite(self):
        """Function analyses bipartite graphs
                Product -> Product (weights)
                Capability -> Capability (weights)

        >>> path = self.params.graph_save_path
        >>> with open(path + 'cG.pickle', 'rb') as f:
        >>>     self.cG = pickle.load(f)

        """
        # cG bipartite projection analysis.
        edge_df_cg = nx.to_pandas_edgelist(self.cG)
        edge_df_cg = edge_df_cg.sample(n=30000).reset_index(drop=True)
        fig = ff.create_distplot([edge_df_cg['weight']],
                                 group_labels=['cG Projection Weights'])
        fig.update_layout(font_family='Arial',
                          title='Bipartite Projection (bG) Weight Distribution',
                          yaxis_title=r"P(w)",
                          xaxis_title=r"w - Weight",
                          # legend_title='Legend',
                          font=dict(size=24))
        fig.write_html(self.params.plotting.path + 'cG_weight_distribution.html')

        counts = dict(Counter(edge_df_cg['weight']))
        loglog_df = \
            pd.DataFrame.from_dict(counts, orient='index').reset_index()
        loglog_df = loglog_df.rename(columns={'index': 'WEIGHT_COUNT',
                                              0: 'FRACTION_OF_WEIGHTS'})

        number_of_edges = edge_df_cg.shape[0]
        loglog_df['FRACTION_OF_WEIGHTS'] = (
                loglog_df['FRACTION_OF_WEIGHTS'] / number_of_edges
        )
        fig = px.scatter(data_frame=loglog_df,
                         x='WEIGHT_COUNT',
                         y='FRACTION_OF_WEIGHTS',
                         log_x=True,
                         log_y=True)
        title = 'Log Log of Weight Distribution for (Product -> Product Graph)'
        fig.update_layout(font_family='Arial',
                          title=title,
                          yaxis_title='Fraction of Edges (log)',
                          xaxis_title='Weight (log)',
                          font=dict(size=24))

        fig.write_html(self.params.plotting.path + 'cG_Weight_log_log.html')

        from scipy.integrate import cumtrapz
        import numpy as np

        x = np.linspace(0, 1000, edge_df_cg['weight'].shape[0])
        cdf = cumtrapz(x=x, y=edge_df_cg['weight'])
        cdf = (cdf / max(cdf))*100

        data = edge_df_cg['weight'].sort_values(ascending=True)
        # fig, ax = plt.subplots(ncols=1)
        plt.figure(figsize=(6, 3.5))
        sns.set_context("paper", font_scale=1.6)
        sns.set_style('ticks')
        sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
        ax = sns.histplot(data,
                          kde=False, color=sns.xkcd_rgb['strong blue'],
                          bins=15, alpha=1, log_scale=(True, True),
                          cumulative=False, linewidth=.8, edgecolor='k',
                          kde_kws={'cumulative': True})
        # frequencies, _ = np.histogram(data)
        # ax.set_yticks(np.arange(0, max(frequencies)/2, 50e3))

        plt.xlabel('Projection Edge Weights (log)')
        ax.set_title(r'Edge weights '
                     '\n'
                     r'Bipartite Projection Weights')
        plt.tight_layout()
        plt.gcf().subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
        plt.ylabel('Frequency (log)')

        # # Second axis for total variation captured
        # ax2 = plt.twinx()
        # ax2.plot(data[1:], cdf)
        # ax2.set_yticks(np.arange(0, 120, 20))
        # ax2.set_ylabel('Proportion CDF (%)')
        plt.savefig(self.params.plotting_path + 'cG_weights_histogram_tex.png',
                    bbox_inches='tight')
        plt.show()

        ########################################################################
        # Analysing the capability -> product bipartite projection weights
        ########################################################################
        capability_product_graph = \
            nx.to_pandas_edgelist(self.capability_product_graph)
        capability_product_graph = (
            capability_product_graph
            .sort_values(by='weight', ascending=False)
            .reset_index(drop=True)
        )

        plt.figure(figsize=(6, 3.5))
        sns.set_context("paper", font_scale=1.8)
        sns.set_style('ticks')
        sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
        sns.histplot(capability_product_graph['weight'],
                     kde=False, color=sns.xkcd_rgb['red'],
                     bins=25, alpha=1, log_scale=True, cumulative=False,
                     linewidth=.8, edgecolor='k',
                     kde_kws={'cumulative': True})
        plt.xlabel('Co-occurrence count (log)')
        plt.title(r'Co-occurrence frequency for (Capability $\rightarrow$ Product)')
        plt.tight_layout()
        plt.gcf().subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
        plt.ylabel('Frequency')
        plt.savefig(self.params.plotting_path + 'capability_product_weights_latex.png',
                    bbox_inches='tight')
        plt.show()

    def create_capability_product_graph(self):
        """
        Creates Capability -> Product Graph
        """
        logger.info('Creating Capability --> Product Weighted Network')
        edge_bunch_list = list()

        for company in tqdm.tqdm(self.companies_all):
            capabilities = \
                [el[1] for el in self.company_capability_graph.edges(company)]
            products = [el[1] for el in self.bG_clean.edges(company)]
            # edge_bunch = tuple(map(lambda x: tuple((x, p) for p in products),
            #                        capabilities))[0]
            #
            edge_bunch_list += list(product(capabilities, products))

        edge_bunch_dict = dict(Counter(edge_bunch_list))
        for k, v in zip(edge_bunch_dict.keys(), edge_bunch_dict.values()):
            # edge_bunch = [(k[0], k[1], v)]
            self.capability_product_graph.add_edge(k[0], k[1], weight=v)

    def create_capability_capability(self):
        """Creates a bipartite production to find complementary Capabilities
        """
        assert nx.is_bipartite(self.company_capability_graph)
        capabilities = list(
            set([el[1] for el in self.company_capability_graph.edges])
            | set([el[0] for el in self.capability_product_graph.edges])
        )
        capability_projection = (
            bipartite.weighted_projected_graph(self.company_capability_graph,
                                               nodes=capabilities)
        )
        # Projection above produces no edges because all weights are 0.
        self.capability_graph = capability_projection

    def create_company_country_links(self) -> None:
        """

        This function reads raw_df from self and creates
        Company -> located_in -> Country

        """
        # Strip the rubbish from the company names
        # self.raw_df['COMPANY_NAME'] = (
        #     self.raw_df['CompanyName']
        #         .apply(lambda x: x.strip("\t").strip("\r\n"))
        # )
        #
        # #
        # self.raw_df['COMPANY_NAME_CLEAN_CO'] = \
        #     self.raw_df['COMPANY_NAME'].apply(lambda x: cleanCompany(x))
        #
        # self.raw_df['COMPANY_NAME_CLEAN_CO_LOWER'] = \
        #     self.raw_df['COMPANY_NAME_CLEAN_CO'].apply(lambda x: x.title())
        #
        # self.raw_df['COMPANY_NAME_CLEAN_CO_LOWER'].head(100)
        # cols_keep = ['Country', 'Continent',
        #              'Quality', 'Remarks', 'COMPANY_NAME_CLEAN_CO_LOWER']
        # self.raw_df_truncated = self.raw_df[cols_keep]
        # data_path = '../data/02_intermediate/raw_df_truncated.parquet'
        # self.raw_df_truncated.to_parquet(data_path, engine='pyarrow')

        cond_1 = \
            self.raw_df['COMPANY_NAME_CLEAN_CO_LOWER'].isin(self.companies_all)
        cond_2 = ~(self.raw_df['Country'].isna())
        cond = cond_1 & cond_2

        self.raw_df = self.raw_df.loc[cond, :]
        col_select = ['COMPANY_NAME_CLEAN_CO_LOWER', 'Country']
        edge_bunch = [(u, v) for u, v in self.raw_df[col_select].values]
        self.company_country_graph.add_edges_from(edge_bunch)
        self.countries_all = \
            list(set([_el[1] for _el in self.company_country_graph.edges]))

    def create_company_qualification_graph(self) -> None:
        """

        This function reads raw_df from self and creates
        Company -> has_cert -> Certification

        """
        cols = ['COMPANY_NAME_CLEAN_CO_LOWER', 'Quality']
        cond_1 = \
            self.raw_df[cols[0]].isin(self.companies_all)
        cond_2 = ~(self.raw_df[cols[1]].isna())
        cond = cond_1 & cond_2
        self.raw_df = self.raw_df.loc[cond, :]
        self.raw_df = self.raw_df.reset_index(drop=True)

        def add_qualification(row):
            company = row[cols[0]]
            company = str(company)
            qualifications = row[cols[1]].split(sep=',')
            return [(company, qual) for qual in qualifications]
        edge_bunches = list(self.raw_df[cols].apply(add_qualification, axis=1))

        edge_bunch = list(chain(*edge_bunches))
        self.company_certification_graph.add_edges_from(edge_bunch)
        self.certifications_all = \
            list(set([_el[1] for _el in self.company_certification_graph.edges]))

    def cut_capability_product_graph(self):
        """
        Remove edges from the capability -> product graph depending on the
        cut threshold provided by the parameters.
        """
        cut_threshold = self.params.capability_product_weight_cut

        capability_product_graph = \
            nx.to_pandas_edgelist(self.capability_product_graph)

        # Define the condition for cutting
        cond = capability_product_graph['weight'] <= cut_threshold
        capability_product_graph = capability_product_graph.loc[cond]

        # Reset the capability_product_graph
        self.capability_product_graph = nx.DiGraph()
        edge_bunch = capability_product_graph[['source', 'target']].values
        self.capability_product_graph.add_edges_from(ebunch_to_add=edge_bunch)

    def save(self, path: str = '../data/02_intermediate/') -> object:
        """
        Creates a new object and saves all graphs into the object.
        """
        # Create all graphs from scratch again
        self.cut_cg()
        self.clean_and_generate_graphs()
        self.create_capability_capability()
        self.create_capability_product_graph()
        self.analyse_bipartite()
        self.cut_capability_product_graph()
        self.create_company_country_links()
        self.create_company_qualification_graph()

        # Pickle self into path provided
        # source, destination
        # with open(self.save_graph_path + 'dataset.pickle', 'wb') as file_path:
        #     logger.info('Saving graphs with the following dimensions:')
        #     logger.info('====================================================')
        #     logger.info(f'cG should have {len(self.cG_clean.edges)} edges')
        #     logger.info(f'bG should have {len(self.bG_clean.edges)} edges')
        #     logger.info(f'G should have {len(self.G_clean.edges)} edges')
        #     logger.info(f'capability_product_graph should have {len(self.capability_product_graph.edges)} edges')
        #     logger.info(f'company_capability_graph should have {len(self.company_capability_graph.edges)} edges')
        #     logger.info(f'{len(self.capabilities_all)} Capabilities')
        #     logger.info(f'{len(self.processes_all)} Processes')
        #     logger.info(f'{len(self.companies_all)} Companies')
        #     logger.info('====================================================')
        #
        #     pickle.dump(copy(self), file_path)
        return copy(self)

    def load(self, from_scratch: bool = False,
             path: str = '../data/02_intermediate/') -> object:
        """Function loads all graphs to memory for use in DGL Dataset creation
        Returns:
            Last reflection of self object in dataset
        """
        if from_scratch:
            return self.save()
        else:
            with open(self.save_graph_path + 'dataset.pickle', 'rb') as \
                    file_path:
                loaded_object = pickle.load(file_path)

            logger.info('Graphs loaded locally with the following dimensions:')
            logger.info('====================================================')
            logger.info(f'cG should have {len(loaded_object.cG_clean.edges)} edges')
            logger.info(f'bG should have {len(loaded_object.bG_clean.edges)} edges')
            logger.info(f'G should have {len(loaded_object.G_clean.edges)} edges')
            logger.info(f'capability_product_graph should have {len(loaded_object.capability_product_graph.edges)} edges')
            logger.info(f'company_capability_graph should have {len(loaded_object.company_capability_graph.edges)} edges')
            logger.info(f'{len(loaded_object.capabilities_all)} Capabilities')
            logger.info(f'{len(loaded_object.processes_all)} Processes')
            logger.info(f'{len(loaded_object.companies_all)} Companies')
            logger.info('====================================================')
            return loaded_object
