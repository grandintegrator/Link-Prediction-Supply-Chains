from torch import cat, ones, zeros
from torch import no_grad
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from typing import Dict, Any
import tqdm


class Evaluator(object):
    def __init__(self, params, model, testing_data_loader):
        self.params = params
        self.model = model
        self.testing_data_loader = testing_data_loader
        self.edge_inference = ('company', 'buys_from', 'company')

    def evaluate(self) -> Dict[str, Any]:
        """Function evaluates the model on all batches and saves all values
        Returns:

        """
        # Turn model into evaluation mode
        self.model.eval()

        # Initialise lists to store evaluation results into.
        scores_all = []
        labels_all = []
        auc_all = []
        auc_pr = []

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
                    pos_score, neg_score = self.model(positive_graph=positive_graph,
                                                      negative_graph=negative_graph,
                                                      blocks=blocks,
                                                      x=input_features)
                    pos_score = pos_score[self.edge_inference]
                    neg_score = neg_score[self.edge_inference]

                    scores_batch = cat([pos_score, neg_score]).detach().numpy()
                    labels_batch = \
                        cat([ones(pos_score.shape[0]), zeros(neg_score.shape[0])])
                    auc_batch = roc_auc_score(labels_batch, scores_batch)
                    pr_batch = average_precision_score(labels_batch, scores_batch)

                    # Add all batch scores/labels/metrics into a large test vector
                    scores_all.extend([scores_batch])
                    labels_all.extend([labels_batch])
                    auc_all.extend([auc_batch])
                    auc_pr.extend([pr_batch])

                # Batch AUC value
                tq.set_postfix({'Batch AUC': f'{auc_batch}'},
                               refresh=False)

            return {'auc_mean': sum(auc_all)/len(auc_all),
                    'auc_all': roc_auc_score(labels_all, scores_all)}
