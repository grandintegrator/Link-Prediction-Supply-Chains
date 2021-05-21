import networkx as nx
import pickle
import tqdm
import pandas as pd
import logging
from copy import copy

# Logger preferences
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig()
logger = logging.getLogger(__name__)


class KnowledgeGraphGenerator(object):
    def __init__(self, path: str = '../data/02_intermediate/'):
        """"
        This class looks at the existing graphs, G, bG, and cG
        couples that information with what we know about capabilities to create

        Returns
        -------
        g (company -> company)
        bG (company -> product)
        company_capability_graph (company -> capability)
        capability_product_graph (capability -> product)

        Graphs based on Bipartite projection (TBD):
        cG (product -> product) ???
        capability_graph (capability -> capability) ???

        """
        ########################################################################
        # Load pickled objects from Edward's analysis
        ########################################################################
        with open(path + 'G.pickle', 'rb') as f:
            self.G = pickle.load(f)

        with open(path + 'cG.pickle', 'rb') as f:
            self.cG = pickle.load(f)

        self.supplier_product_ds = \
            pd.read_parquet('../data/01_raw/supplier_product_df.parquet')

        ########################################################################
        # Create empty graph objects to store data into
        ########################################################################
        self.bG = nx.DiGraph()
        self.bG_clean = nx.DiGraph()
        self.G_clean = nx.DiGraph()
        self.cG_clean = nx.DiGraph()
        self.company_capability_graph = nx.DiGraph()
        self.capability_product_graph = nx.DiGraph()
        self.capability_graph = nx.DiGraph()  # TODO

        self.processes_all = None
        self.companies_all = None

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

    def create_bg_clean(self):
        logger.info('Creating bG_clean graph and saving all capabilities.')
        ########################################################################
        # Fix bG overlap issue (companies -> products)
        ########################################################################
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
        for edge in tqdm.tqdm(self.cG.edges):
            p1 = edge[0].title()
            p2 = edge[1].title()

            if (p1 in capabilities_found) and (p2 not in capabilities_found):
                self.capability_product_graph.add_edge(u_of_edge=p1,
                                                       v_of_edge=p2)
            elif (p2 in capabilities_found) and (p1 not in capabilities_found):
                self.capability_product_graph.add_edge(u_of_edge=p2,
                                                       v_of_edge=p1)

        ########################################################################
        # Crease Company-Capability graph - (Company -> Capability)
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
            set([el[0].title() for el in self.G_clean.edges])
        )

        ########################################################################
        # Fix cG by getting rid of all product -> products as capabilities
        # len(self.cG.edges)
        #           Goes from 472,978 ---> 353,060 (25% reduction)
        ########################################################################
        cg_edge_df = nx.to_pandas_edgelist(self.cG)
        cg_edge_df = \
            cg_edge_df.drop(['weight'], axis=1).apply(lambda x: x.str.title())
        cond = (
            cg_edge_df['source'].isin(self.capabilities_all)
            | cg_edge_df['source'].isin(self.companies_all)
            | cg_edge_df['target'].isin(self.capabilities_all)
            | cg_edge_df['target'].isin(self.companies_all)
        )
        cg_edge_df = cg_edge_df.loc[~cond]
        self.cG_clean.add_edges_from([(u, v) for u, v in cg_edge_df.values])

    def save(self, path: str = '../data/02_intermediate/') -> object:
        """
        Creates a new object and saves all graphs into the object.
        """
        # Create all graphs from scratch again
        self.clean_and_generate_graphs()

        # Pickle self into path provided
        # source, destination
        with open(path + 'dataset.pickle', 'wb') as file_path:

            logger.info('Saving graphs with the following dimensions:')
            logger.info('====================================================')
            logger.info(f'cG should have {len(self.cG_clean.edges)} edges')
            logger.info(f'bG should have {len(self.bG_clean.edges)} edges')
            logger.info(f'G should have {len(self.G_clean.edges)} edges')
            logger.info(f'capability_product_graph should have {len(self.capability_product_graph.edges)} edges')
            logger.info(f'company_capability_graph should have {len(self.company_capability_graph.edges)} edges')
            logger.info(f'{len(self.capabilities_all)} Capabilities')
            logger.info(f'{len(self.processes_all)} Processes')
            logger.info(f'{len(self.companies_all)} Companies')
            logger.info('====================================================')

            pickle.dump(copy(self), file_path)
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
            with open(path + 'dataset.pickle', 'rb') as file_path:
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
