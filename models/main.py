"""Script to run the baselines."""
import argparse
import importlib
import numpy as np
import os
import sys
import logging
import random
import torch
import wandb

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server

import metrics.writer as metrics_writer

from utils.args import parse_args
from utils.logging import setuplogger
from utils.data import read_data


STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

def main():

    args = parse_args()

    # Suppress logging
    setuplogger(args)

    # initialize wandb
    if args.enable_wandb:
        wandb.init(
                project=f"{args.wandb_project}-{args.dataset}-{args.model}",
                config=args,
                name=f"{args.wandb_run}",
            )
        logging.info("[-] finishing initing wandb.")

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    torch.manual_seed(123 + args.seed)
    torch.cuda.manual_seed(123 + args.seed)

    model_path = f'{args.dataset}/{args.model}.py'
    if not os.path.exists(model_path):
        logging.error('Please specify a valid dataset and a valid model.')
    model_path = f'{args.dataset}.{args.model}'
    
    logging.info(f'############################## {model_path} ##############################')
    mod = importlib.import_module(model_path)
    ModelCls = getattr(mod, 'ClientModel')

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]


    # Create 2 models
    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    # Create sever model
    global_model = ModelCls(args.seed, *model_params)

    # Create server
    server = Server(args, global_model)

    # Create client model
    client_model = ModelCls(args.seed, *model_params).cuda()

    # Create clients
    clients = setup_clients(args.dataset, client_model, args.use_val_set)
    client_ids, client_groups, client_num_samples = server.get_clients_info(clients)
    logging.info('Clients in Total: %d' % len(clients))

    # Initial status
    logging.info('--- Random Initialization ---')
    stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)
    print_stats(0, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)

    # Simulate training
    for i in range(num_rounds):
        logging.info('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

        # Select clients to train this round
        server.select_clients(i, online(clients), num_clients=clients_per_round)
        c_ids, c_groups, c_num_samples = server.get_clients_info(server.selected_clients)

        # Simulate server model training on selected clients' data
        sys_metrics = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch)
        sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples)
        
        # Update server model
        server.update_model()

        # Test model
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            print_stats(i + 1, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)
    
    # Save server model
    ckpt_path = os.path.join('checkpoints', args.dataset)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server.save_model(os.path.join(ckpt_path, '{}.ckpt'.format(args.model)))
    print('Model saved in path: %s' % save_path)


def online(clients):
    """We assume all users are always online."""
    return clients


def create_clients(users, groups, train_data, test_data, model):
    if len(groups) == 0:
        groups = [[] for _ in users]
    clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
    return clients


def setup_clients(dataset, model=None, use_val_set=False):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    eval_set = 'test' if not use_val_set else 'val'
    train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset, 'data', eval_set)

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    clients = create_clients(users, groups, train_data, test_data, model)

    return clients


def get_stat_writer_function(ids, groups, num_samples, args):

    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, partition, args.metrics_dir, '{}_{}'.format(args.metrics_name, 'stat'))

    return writer_fn


def get_sys_writer_function(args):

    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, 'train', args.metrics_dir, '{}_{}'.format(args.metrics_name, 'sys'))

    return writer_fn


def print_stats(
    num_round, server, clients, num_samples, args, writer, use_val_set):
    
    train_stat_metrics = server.test_model(clients, set_to_use='train')
    print_metrics(num_round, train_stat_metrics, num_samples, prefix='train_', enable_wandb=args.enable_wandb)
    writer(num_round, train_stat_metrics, 'train')

    eval_set = 'test' if not use_val_set else 'val'
    test_stat_metrics = server.test_model(clients, set_to_use=eval_set)
    print_metrics(num_round, test_stat_metrics, num_samples, prefix='{}_'.format(eval_set), enable_wandb=args.enable_wandb)
    writer(num_round, test_stat_metrics, eval_set)


def print_metrics(num_round, metrics, weights, prefix='', enable_wandb=False):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    to_ret = None
    metrics_log = {}
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        logging.info(
            '%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
            % ( prefix + metric,
                np.average(ordered_metric, weights=ordered_weights),
                np.percentile(ordered_metric, 10),
                np.percentile(ordered_metric, 50),
                np.percentile(ordered_metric, 90)))

        metrics_log[prefix + metric] = np.average(ordered_metric, weights=ordered_weights)

    if enable_wandb:
        wandb.log(
            metrics_log, step = num_round
        )


if __name__ == '__main__':
    main()
