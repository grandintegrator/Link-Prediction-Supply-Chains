import argparse
import random
import torch
import logging
import numpy as np
import yaml
from managers.trainer import Trainer
from managers.evaluator import Evaluator
from warnings import simplefilter
from ingestion.dataloader import SCDataLoader
from utils import save_best_metrics
from box import Box
from utils import create_model


################################################################################
# Deterministic behaviour for experimentation
################################################################################
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochastic behaviour") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# # Device configuration
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(run_args) -> None:
    simplefilter(action='ignore', category=UserWarning)

    with open('config/config.yml', 'r') as yaml_file:
        config = Box(yaml.safe_load(yaml_file), default_box=True,
                     default_box_attr=None)

    logging.info('============================================================')
    logging.info(f'The following run args have been run \n {config.to_dict()}')
    logging.info('============================================================')

    data_loader = SCDataLoader(params=config)
    train_loader = data_loader.get_training_dataloader()

    graph_model = create_model(params=config,
                               graph_edge_types=data_loader.edge_types)

    trainer = Trainer(params=config, model=graph_model,
                      train_data_loader=train_loader)
    logging.info('Starting training of model...')
    trainer.train()
    save_best_metrics(path=config.plotting.path)

    logging.info('Saved Training results. Evaluating model on testing data...')
    test_loader = data_loader.get_test_data_loader()
    evaluator = Evaluator(params=config, model=trainer.model,
                          testing_data_loader=test_loader)
    metrics_test = evaluator.evaluate()
    logging.info('============================================================')
    logging.info(f'After {config.modelling.num_epochs} epochs \n')
    logging.info(f"Got an AUC of  {metrics_test['auc_mean']}")
    logging.info('============================================================')


if __name__ == '__main__':
    # Parser arguments
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Supply Chain KG Completion')

    # Parser arguments for run...
    parser.add_argument("--device", type=str, default='cpu',
                        help="Did you finally get a GPU?")

    # initialize_experiment(params, __file__)
    args = parser.parse_args()
    main(args)
