import logging
import tqdm
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import cat, ones, zeros
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from typing import Dict, Any
from model.dgl.StochasticRGCN import ScorePredictor


class Trainer(object):
    def __init__(self, params, model, train_data_loader):
        self.params = params
        self.model = model
        self.train_data_loader = train_data_loader
        self.edge_inference = ('company', 'buys_from', 'company')
        self.predictor = ScorePredictor().to(params.device)

        # Fixing parameter types because Box doesn't do this naturally.
        self.params.net.lr = float(self.params.net.lr)
        self.params.net.l2_regularisation = \
            float(self.params.net.l2_regularisation)

        # Could probably turn this into a function if we want to try others
        if params.net.optimiser == 'SGD':
            self.opt = optim.SGD(self.model.parameters(), lr=self.params.net.lr,
                                 momentum=self.params.net.momentum,
                                 weight_decay=self.params.net.l2_regularisation)
        elif params.net.optimiser == 'Adam':
            self.opt = optim.Adam(self.model.parameters(),
                                  lr=self.params.net.lr,
                                  weight_decay=self.params.net.l2_regularisation)

    def compute_loss(self, pos_score, neg_score):
        # For computing the pos and negative score just for the inference edge
        # TODO: Extend this for multiple edge types with cross_entropy loss
        # TODO: Add in option for impact of losses
        assert pos_score.keys() == neg_score.keys()
        all_losses = []
        for given_type in pos_score.keys():
            pos_score_edge = pos_score[given_type]
            neg_score_edge = neg_score[given_type]
            n = pos_score_edge.shape[0]
            if n == 0:
                continue
            if self.params.net.loss == 'margin':

                margin_loss = (
                    (neg_score_edge.view(n, -1) -
                     pos_score_edge.view(n, -1) + 1)
                        .clamp(min=0).mean()
                )
                all_losses.append(margin_loss)
            elif self.params.net.loss == 'binary_cross_entropy':
                scores = cat([pos_score_edge, neg_score_edge])
                labels = \
                    cat([ones(pos_score_edge.shape[0]),
                         zeros(neg_score_edge.shape[0])])
                scores = scores.view(len(scores), -1).mean(dim=1)  # Fixing dims
                binary_cross_entropy = \
                    F.binary_cross_entropy_with_logits(scores, labels)
                all_losses.append(binary_cross_entropy)
        return torch.stack(all_losses, dim=0).mean()

    @staticmethod
    def compute_train_auc_ap(pos_score, neg_score) -> Dict[str, Any]:
        # TODO: Extend this for multiple edge types - multi-class accuracy
        #   or 1 vs All?
        # Compute the AUC per type
        auc_dict = {}
        ap_dict = {}
        for given_edge_type in pos_score.keys():
            pos_score_edge = pos_score[given_edge_type]
            neg_score_edge = neg_score[given_edge_type]
            scores = (
                cat([pos_score_edge, neg_score_edge]).detach().numpy()
            )
            labels = cat(
                [ones(pos_score_edge.shape[0]),
                 zeros(neg_score_edge.shape[0])]).detach().numpy()
            auc_dict['AUC ' + given_edge_type[1]] = \
                roc_auc_score(labels, scores)
            ap_dict['AP ' + given_edge_type[1]] = \
                average_precision_score(labels, scores)
        return {'AUC_DICT': auc_dict, 'AP_DICT': ap_dict}

    @staticmethod
    def log_results(step, loss, auc_dict, auc_pr_dict, log_freq) -> None:
        if (step + 1) % log_freq == 0:

            wandb.log({"epoch": step, "Training loss": loss}, step=step)

            # Logging of AUC values for all edge types in the prediction
            for auc_edge in auc_dict.keys():
                wandb.log({"epoch": step,
                           "Training " + auc_edge: auc_dict[auc_edge]},
                          step=step)
            for auc_pr_edge in auc_pr_dict.keys():
                wandb.log({"epoch": step,
                           "Training " + auc_pr_edge: auc_pr_dict[auc_pr_edge]},
                          step=step)

    def train_epoch(self):
        with tqdm.tqdm(self.train_data_loader) as tq:
            for step, (input_nodes, positive_graph, negative_graph,
                       blocks) in enumerate(tq):
                # For transferring to CUDA when I finally get a GPU
                if self.params.modelling.device == 'gpu':
                    blocks = [b.to(torch.device('cuda')) for b in blocks]
                    positive_graph = positive_graph.to(torch.device('cuda'))
                    negative_graph = negative_graph.to(torch.device('cuda'))

                # Need to ensure that all node types have been captured
                if any([b.num_edges(edge_type) == 0 for b in blocks
                        for edge_type in blocks[0].etypes]):
                    # Jump to next mini-batch because this one is invalid.
                    # logging.info('Sampled bad training batch')
                    continue

                input_features = blocks[0].srcdata['feature']
                pos_score, neg_score = self.model(positive_graph=positive_graph,
                                                  negative_graph=negative_graph,
                                                  blocks=blocks,
                                                  x=input_features)
                loss = self.compute_loss(pos_score, neg_score)

                # <---: Back Prop :)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # # Logging
                if (step + 1) % self.params.modelling.log_freq == 0:
                    # Compute some training set statistics
                    auc_pr_dicts = self.compute_train_auc_ap(pos_score,
                                                             neg_score)
                    self.log_results(step, loss, auc_pr_dicts['AUC_DICT'],
                                     auc_pr_dicts['AP_DICT'],
                                     self.params.modelling.log_freq)

                tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)
                # Break if number of epochs has been satisfied
                if step == self.params.modelling.num_epochs:
                    break

    def train(self):
        # wandb login --relogin of you would like to log data into W&B
        if self.params.stream_wandb:
            wandb.init()
            wandb.watch(self.model, self.compute_loss, log="all",
                        log_freq=self.params.modelling.log_freq)
        for _ in range(1):
            # Put model into training mode
            self.model.train()
            self.train_epoch()

    def __repr__(self):
        return "Training manager class"
