import tqdm
import wandb
import torch.optim as optim
import torch.nn.functional as F
from torch import cat, ones, zeros
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from typing import List


class Trainer(object):
    def __init__(self, params, model, train_data_loader):
        self.params = params
        self.model = model
        self.train_data_loader = train_data_loader
        self.edge_inference = ('company', 'buys_from', 'company')

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
        pos_score = pos_score[self.edge_inference]
        neg_score = neg_score[self.edge_inference]
        n = pos_score.shape[0]

        if self.params.net.loss == 'margin':
            margin_loss = (
                (neg_score.view(n, -1) - pos_score.view(n, -1) + 1)
                    .clamp(min=0).mean()
            )
            return margin_loss
        elif self.params.net.loss == 'binary_cross_entropy':
            scores = cat([pos_score, neg_score])
            labels = cat([ones(pos_score.shape[0]), zeros(neg_score.shape[0])])
            scores = scores.view(len(scores), -1).mean(dim=1)  # Fixing dims
            return F.binary_cross_entropy_with_logits(scores, labels)

    def compute_train_auc_ap(self, pos_score, neg_score) -> List:
        # TODO: Extend this for multiple edge types - multi-class accuracy
        #   or 1 vs All?
        pos_score = pos_score[self.edge_inference]
        neg_score = neg_score[self.edge_inference]
        scores = (
            cat([pos_score, neg_score]).detach().numpy()
        )
        labels = cat(
            [ones(pos_score.shape[0]),
             zeros(neg_score.shape[0])]).detach().numpy()
        return [roc_auc_score(labels, scores),
                average_precision_score(labels, scores)]

    @staticmethod
    def log_results(step, loss, auc, auc_pr, log_freq) -> None:
        if (step + 1) % log_freq == 0:
            wandb.log({"epoch": step, "loss": loss}, step=step)
            wandb.log({"epoch": step, "auc": auc}, step=step)
            wandb.log({"epoch": step, "av_precision": auc_pr},
                      step=step)

    def train_epoch(self):
        with tqdm.tqdm(self.train_data_loader) as tq:
            for step, (input_nodes, positive_graph, negative_graph,
                       blocks) in enumerate(tq):
                # For transferring to CUDA when I finally get a GPU
                # self.params.modelling.device == 'gpu':
                #     blocks = [b.to(torch.device('cuda')) for b in blocks]
                #     positive_graph = positive_graph.to(torch.device('cuda'))
                #     negative_graph = negative_graph.to(torch.device('cuda'))

                # Need to ensure that all node types have been captured
                if any([b.num_edges(edge_type) == 0 for b in blocks
                        for edge_type in blocks[0].etypes]):
                    # Jump to next mini-batch because this one is invalid.
                    continue

                input_features = blocks[0].srcdata['feature']
                pos_score, neg_score = self.model(positive_graph=positive_graph,
                                                  negative_graph=negative_graph,
                                                  blocks=blocks,
                                                  x=input_features)
                loss = self.compute_loss(pos_score, neg_score)

                # <--- Back Prop :)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # Logging
                if (step + 1) % self.params.modelling.log_freq == 0:
                    # Compute some training set statistics
                    auc, auc_pr = self.compute_train_auc_ap(pos_score,
                                                            neg_score)
                    self.log_results(step, loss, auc, auc_pr,
                                     self.params.modelling.log_freq)

                tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)
                # Break if number of epochs has been satisfied
                if step == self.params.modelling.num_epochs:
                    break

    def train(self):
        # wandb login --relogin of you would like to log data into W&B
        wandb.init()
        wandb.watch(self.model, self.compute_loss, log="all",
                    log_freq=self.params.modelling.log_freq)
        for _ in range(1):
            # self.model.train()
            self.train_epoch()

    def __repr__(self):
        return "Training manager class"