from cleanco import prepare_terms, basename
from datetime import datetime
import random
from tqdm import tqdm
import networkx as nx
import numpy as np
from sklearn.metrics import (
    classification_report,
    log_loss,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

import pandas as pd
import matplotlib.pyplot as plt
import traceback
from _thread import start_new_thread
from functools import wraps
from typing import Dict, Any
import torch
from torch.multiprocessing import Queue
import dgl
import logging
from model.dgl.StochasticRGCN import Model
import wandb


def save_best_metrics(path: str, training_results: bool = True,
                      validation_results: bool = False,
                      testing_results: bool = False) -> None:
    """Function saves the latest run from the weights and biases logs
    """
    save_string = ""
    api = wandb.Api()
    if training_results:
        save_string = 'training'
        runs = api.runs('Link-Prediction-Supply-Chains')
    elif validation_results:
        save_string = 'valid'
        runs = api.runs('Validation')
    elif testing_results:
        save_string = 'testing'
        runs = api.runs('Testing')
    # Get names from runs API if there is an AUC or AP in the log name
    all_runs = runs.objects
    latest_run = all_runs[0]

    # auc_ap_names = (
    #     [name for name in runs.objects[0].summary.keys() if
    #      ('Training AUC' in name) or ('Training AP' in name)]
    # )

    auc_ap_names = ['Validation AUC makes_product',
                    'Validation AP has_capability',
                    'Validation AP has_cert',
                    'Validation AUC has_cert',
                    'Validation AP located_in',
                    'Validation AP buys_from',
                    'Validation AP makes_product',
                    'Validation AUC complimentary_product_to',
                    'Validation AP complimentary_product_to',
                    'Validation AUC located_in',
                    'Validation AP capability_produces',
                    'Validation AUC has_capability',
                    'Validation AUC buys_from',
                    'Validation AUC capability_produces']

    run_summary_dict = latest_run.summary
    run_summary_filtered = (
        {auc_ap: run_summary_dict[auc_ap] for auc_ap in auc_ap_names}
    )
    # Turn the dictionary into a pandas dataframe for saving into results.
    summary_dictionary = (
        pd.DataFrame(run_summary_filtered, index=['Best Training Value']).T
    )

    summary_dictionary.to_csv(path + 'wandb-final-' + latest_run.name +
                              save_string + '.csv')


def create_model(params, graph_edge_types):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # in_features, hidden_features, out_features, num_classes,
    # etypes):
    model = Model(in_features=params.num_node_features,
                  hidden_features=params.num_hidden_graph_layers,
                  out_features=params.num_node_features,
                  num_classes=params.num_classes,
                  etypes=graph_edge_types)
    return model


def getTime():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


# return default value if error
def get(function, arg, default):
    try:
        return function(arg)
    except Exception as e:
        return default


"""
PREPROCESSING
"""


def stripCompany(business_name, n=2):
    # clean n times because some companies have multiple legal statuses

    # clean terms
    terms = prepare_terms()
    newName = basename(business_name, terms, prefix=True, middle=False, suffix=True)

    if n > 0:
        return stripCompany(newName, n - 1)
    else:
        return newName


def cleanCompany(name):
    stripName = stripCompany(name)
    stripName = stripName.upper()
    return stripName


def cleanProduct(text, removeWords):
    for word in removeWords:
        text = text.replace(word, "")

    return text


# take in string list and return list
def splitMarklinesList(textList):
    if pd.isna(textList):
        return []
    else:
        textList = textList.strip("\t").strip("\r\n")
        textList = str(textList).split("\r\n")
        textList = [_list.upper() for _list in textList]
        return textList


# TODO: DEPRECATED
# take in string list and return list
def splitCustomers(textList):
    if pd.isna(textList):
        return []
    else:
        textList = textList.strip("\t").strip("\r\n")
        textList = str(textList).split("\r\n")
        textList = [_list.upper() for _list in textList]
        return textList


"""
SPLIT TRAIN-TEST
"""


def gen_train_test_extra(G, extraEdges):
    # [RANDOM SAMPLING] GATHER TRAIN AND TEST DATA
    # (node1, node2, isEdgeExist)
    trainDataEdges = []
    testDataEdges = []

    count = 0
    # sample edges
    edges = list(G.edges())
    M = len(edges)

    # KEY: SHUFFLE THE TRAINING/TESTING EDGES
    random.shuffle(edges)

    for node1, node2 in tqdm(edges):
        if count == M:
            break

        y = G.has_edge(node1, node2)

        if count < int(M / 2):
            trainDataEdges.append([node1, node2, y])
        else:
            testDataEdges.append([node1, node2, y])

        count += 1

    # sample non-edges
    count = 0
    testM = M  # 3*M
    for node1, node2 in tqdm(nx.non_edges(G)):
        # only keep M non-edges
        if count == testM:
            break

        y = G.has_edge(node1, node2)

        # selectively sample non-edge
        if np.random.uniform() > 0.9:

            if count < int(testM / 2):
                trainDataEdges.append([node1, node2, y])
            else:
                testDataEdges.append([node1, node2, y])

            count += 1

    # sample non-edges
    count = 0
    # KEY: SHUFFLE THE TRAINING/TESTING EDGES
    random.shuffle(edges)

    for node1, neighNode in tqdm(edges):
        # only keep M non-edges
        if count == M:
            break

        # get neighbours from both sides
        N2 = list(G.neighbors(neighNode))
        random.shuffle(N2)

        # find neighbors of N2 that are not connected to node1
        node2 = None
        for neigh in N2:
            if (not G.has_edge(node1, neigh)) & (node1 != neigh):
                node2 = neigh
                break

        if node2 == None:
            continue

        y = G.has_edge(node1, node2)

        if count < int(M / 2):
            trainDataEdges.append([node1, node2, y])
        else:
            testDataEdges.append([node1, node2, y])

        count += 1

    # extraEdges
    extraDataEdges = []
    for node1, node2 in tqdm(extraEdges):
        y = G.has_edge(node1, node2)

        extraDataEdges.append([node1, node2, y])

    return trainDataEdges, testDataEdges, extraDataEdges


"""
RESULT ANALYSIS
"""


# evaluate dataframe results
def evaluate(__df__, index, thresh=0.5):
    try:
        ## PURE HISTOGRAM
        plt.figure(figsize=(10, 5))
        for lab, scoreDF in __df__.groupby("label"):
            c = "b" if lab == 1 else "r"
            name = "edge" if lab == 1 else "non-edge"
            plt.hist(scoreDF[index], color=c, alpha=0.3, label=name)

        plt.legend()
        plt.show()

        ## CLASSIFICATION
        predThres = __df__[index].apply(lambda x: x > thresh)
        print(classification_report(__df__["label"], predThres))

        ## AUC-ROC
        y_true = __df__["label"]
        y_score = __df__[index]

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figheight(10)
        fig.set_figwidth(20)
        fpr, tpr, thresholds = precision_recall_curve(y_true, y_score)
        ax1.plot(fpr, tpr)
        ax1.set_title("Precision-Recall Curve")
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        ax2.plot(fpr, tpr)
        ax2.set_title("ROC Curve, AUC = {:.2f}".format(roc_auc_score(y_true, y_score)))
        plt.show()
    except Exception as e:
        print(e)


"""
Compare All Results
"""


# evaluation function
def plotCurves(__df__, columns):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(10)
    fig.set_figwidth(20)

    for index in columns:
        ## AUC-ROC
        y_true = __df__["label"]
        y_score = __df__[index]

        auc = roc_auc_score(y_true, y_score)

        fpr, tpr, thresholds = precision_recall_curve(y_true, y_score)
        #         ax1.plot(fpr, tpr, label=index)
        ax1.plot(fpr, tpr, label="{}-{:.2f}".format(index, average_precision_score(y_true, y_score)))
        ax1.set_title("Precision-Recall Curve")
        ax1.legend()
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        ax2.plot(fpr, tpr)
        ax2.plot(fpr, tpr, label="{}-{:.2f}".format(index, auc))
        ax2.set_title("ROC Curve")
        ax2.legend()

    plt.show()


def initialize_experiment(params):
    # Logging options
    logger = logging.getLogger(__name__)
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig()
