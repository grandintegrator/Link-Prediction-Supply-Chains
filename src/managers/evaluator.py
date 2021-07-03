import numpy as np
import pandas as pd
from torch import cat, ones, zeros
from torch import no_grad
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from typing import Dict, Any
from torch.nn import Sigmoid

import tqdm
import wandb
import logging
import uuid


class Evaluator(object):
    def __init__(self, params, model, testing_data_loader):
        self.params = params
        self.model = model
        self.testing_data_loader = testing_data_loader
        self.sigmoid = Sigmoid()
        self._scores_all = np.array([])
        self._labels_all = np.array([])
        self._link_type_list = list()
        self._batch_id = list()
        self.validation_frame = pd.DataFrame()

    @staticmethod
    def log_test_results(step, auc_dict, auc_pr_dict, log_freq,
                         eval_type) -> None:
        """Logs testing results to w&b
        """
        if (step + 1) % log_freq == 0:
            # wandb.log({"epoch": step, "Test AUC (Batch)": auc_batch}, step=step)
            # wandb.log({"epoch": step, "Test AUC-PR (Batch)": pr_batch},
            #           step=step)
            # Logging of AUC values for all edge types in the prediction
            for auc_edge in auc_dict.keys():
                wandb.log({"epoch": step,
                           eval_type + " " + auc_edge: auc_dict[auc_edge]},
                          step=step)
            for auc_pr_edge in auc_pr_dict.keys():
                wandb.log({"epoch": step,
                           eval_type + " " + auc_pr_edge: auc_pr_dict[auc_pr_edge]},
                          step=step)

    def compute_testing_auc_ap(self, sigmoid, pos_score, neg_score) -> Dict[str, Any]:
        # Compute the AUC per type
        auc_dict = {}
        ap_dict = {}
        for given_edge_type in pos_score.keys():
            pos_score_edge = sigmoid(pos_score[given_edge_type])
            neg_score_edge = sigmoid(neg_score[given_edge_type])
            if pos_score_edge.shape[0] == 0:
                continue
            scores = (
                cat([pos_score_edge, neg_score_edge]).detach().numpy()
            )
            labels = cat(
                [ones(pos_score_edge.shape[0]),
                 zeros(neg_score_edge.shape[0])]).detach().numpy()

            if self.params.save_validation_frame:
                self._scores_all = np.append(self._scores_all, scores.squeeze())
                self._labels_all = np.append(self._labels_all,
                                             labels.squeeze().astype('int'))
                self._link_type_list.extend(
                    [given_edge_type[1]] * len(scores.squeeze())
                )
                self._batch_id.extend([uuid.uuid4()] * len(scores.squeeze()))

            auc_dict['AUC ' + given_edge_type[1]] = \
                roc_auc_score(labels, scores)
            ap_dict['AP ' + given_edge_type[1]] = \
                average_precision_score(labels, scores)
        return {'AUC_DICT': auc_dict, 'AP_DICT': ap_dict}

    def store_validation_frame(self) -> None:
        logging.info('Saving validation outputs and labels.')
        self.validation_frame = (
            pd.DataFrame({'MODEL_SCORE': self._scores_all,
                          'LABELS': self._labels_all,
                          'LINK_TYPE': self._link_type_list,
                          'BATCH_ID': self._batch_id})
        )
        self.validation_frame['BATCH_ID'] = (
            self.validation_frame['BATCH_ID'].astype('str')
        )
        store_path = self.params.model_save_path + 'validation_frame.parquet'
        self.validation_frame.to_parquet(store_path, compression='gzip',
                                         engine='pyarrow')

    def evaluate(self) -> None:
        """Function evaluates the model on all batches and saves all values
        Returns:

        """
        # Turn model into evaluation mode
        self.model.eval()

        if self.params.stream_wandb:
            wandb.init(config=dict(self.params), project=self.params.eval_type)
            wandb.watch(self.model, self.log_test_results, log="all",
                        log_freq=self.params.log_freq)

        with tqdm.tqdm(self.testing_data_loader) as tq:
            for step, (input_nodes, positive_graph, negative_graph,
                       blocks) in enumerate(tq):

                # Need to ensure that all node types have been captured
                if any([b.num_edges(edge_type) == 0 for b in blocks
                        for edge_type in blocks[0].etypes]):
                    # Jump to next mini-batch because this one is invalid.
                    continue
                with no_grad():
                    input_features = blocks[0].srcdata['feature']

                    # 🔜 Forward pass through the network.
                    pos_score, neg_score = self.model(positive_graph=positive_graph,
                                                      negative_graph=negative_graph,
                                                      blocks=blocks,
                                                      x=input_features)

                    auc_pr_dicts = self.compute_testing_auc_ap(self.sigmoid,
                                                               pos_score,
                                                               neg_score)
                    if self.params.stream_wandb:
                        self.log_test_results(step, auc_pr_dicts['AUC_DICT'],
                                              auc_pr_dicts['AP_DICT'],
                                              self.params.log_freq,
                                              self.params.eval_type)
                    # pos_score = pos_score[self.edge_inference]
                    # neg_score = neg_score[self.edge_inference]
                    #
                    # scores_batch = (
                    #     cat([pos_score, neg_score]).detach().numpy()
                    # ).squeeze()
                    # labels_batch = (
                    #     cat([ones(pos_score.shape[0]),
                    #          zeros(neg_score.shape[0])]).detach().numpy()
                    # )
                    #
                    # auc_batch = roc_auc_score(labels_batch, scores_batch)
                    # pr_batch = \
                    #     average_precision_score(labels_batch, scores_batch)
                    #
                    # # Add all batch scores/labels/metrics into a large test vector
                    # scores_all.extend([scores_batch])
                    # labels_all.extend([labels_batch])
                    # auc_all.extend([auc_batch])
                    # auc_pr.extend([pr_batch])

                    # Add results to W & B
                    # self.log_test_results(auc_batch, pr_batch, step,
                    #                       self.params.testing.log_freq)
                    #
                    # # TQ logging
                    # tq.set_postfix({'Batch AUC': f'{auc_batch:.3f}'},
                    #                refresh=False)

            # print(auc_all)
            # return {'auc_mean': sum(auc_all)/len(auc_all),
            #         'auc_all': roc_auc_score(labels_all, scores_all)}
        self.store_validation_frame()
