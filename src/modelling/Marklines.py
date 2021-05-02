import pandas as pd
from tqdm import tqdm
from cleanco import prepare_terms, basename
from datetime import datetime
import networkx as nx

from utils import cleanCompany, splitCustomers, splitMarklinesList, stripCompany, getTime, cleanProduct
from utils import get

import numpy as np

"""
MARKLINES DATABASE
"""


class Marklines:
    def __init__(self, filename, checkpointed=False, checkpointFilename=None):
        self.filename = filename
        self.rawdf = pd.read_excel(filename)

        # if data has been processed before
        if checkpointed == False:
            self.edgedf, self.supplierProductdf, self.companyNameMapping = self.createEdges()
        else:
            self.edgedf = pd.read_csv(checkpointFilename)
            self.supplierProductdf = None

        # generate various graphs
        self.G, self.aG, self.bG, self.cG = self.createGraphs()

    def createEdges(self):
        print(getTime(), "extracting customer and product list")

        # generate two edgedf: supplier network and supplier-product network
        supplierNetworkMaps = []
        supplierProductMaps = []

        # keep track of cleaned company name mapping
        companyNameMapping = {}

        for _, row in tqdm(list(self.rawdf.iterrows())):
            # get focal company name
            company = splitMarklinesList(row["CompanyName"])[0]
            cleanCompanyName = cleanCompany(company)

            # don't include blank company
            if ((cleanCompanyName == "") | (cleanCompanyName is np.nan)):
                continue

            # keep track of cleaned name mapping
            companyNameMapping[company] = cleanCompanyName

            # get customers' name
            customers = splitMarklinesList(row["Customers"])
            for customer in customers:
                cleanCustomerName = cleanCompany(customer)
                # keep track of cleaned name mapping
                companyNameMapping[customer] = cleanCustomerName

                # don't include blank company
                if ((cleanCustomerName == "") | (cleanCustomerName is np.nan)):
                    continue

                supplierNetworkMaps.append([cleanCustomerName, cleanCompanyName])

            # get products' name
            removeWords = [
                '<SPAN STYLE="COLOR: RED;">',
                '</SPAN>'
            ]
            products = splitMarklinesList(row["Products"])
            for product in products:
                cleanProductName = cleanProduct(product, removeWords)
                supplierProductMaps.append([cleanCompanyName, cleanProductName])

        print(getTime(), "creating edges dataframe")
        supplierNetworkMapsDF = pd.DataFrame(supplierNetworkMaps, columns=["companyName", "supplierName"])
        supplierProductMapsDF = pd.DataFrame(supplierProductMaps, columns=["companyName", "product"])

        return supplierNetworkMapsDF, supplierProductMapsDF, companyNameMapping

    ## TODO: pickle the whole object instead of saving individual dataframes
    def checkpointEdgeDF(self, filename):
        self.edgedf.to_csv(filename, index=False)

    # based on Brintrup et al (2018) SNLP paper
    # generate various graphs
    def createGraphs(self):
        # create undirected supply relationship
        G = nx.Graph()
        G.add_edges_from(self.edgedf[["companyName", "supplierName"]].values)

        # create directed supply relationship
        aG = nx.DiGraph()
        aG.add_edges_from(self.edgedf[["supplierName", "companyName"]].values)

        # create supplier-product relationship
        bG = nx.Graph()
        bG.add_edges_from(self.supplierProductdf.values)

        # create product outsourcing graph
        cG = nx.Graph()
        # consider every edge in the supplier graph
        for node1, node2 in tqdm(list(aG.edges())):

            # get products
            prodsNode1 = list(get(bG.neighbors, node1, []))
            prodsNode2 = list(get(bG.neighbors, node2, []))

            # add product weights for every common products used in supplier-buyer
            for prod1 in prodsNode1:
                for prod2 in prodsNode2:
                    if prod2 == prod1:
                        continue

                    # check if edge exists
                    if not cG.has_edge(prod1, prod2):
                        cG.add_edge(prod1, prod2, weight=1)
                    # if yes add weight
                    else:
                        cG[prod1][prod2]["weight"] += 1

        return G, aG, bG, cG