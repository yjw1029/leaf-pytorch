import argparse

DATASETS = ['sent140', 'femnist', 'shakespeare', 'celeba', 'synthetic', 'reddit']
SIM_TIMES = ['small', 'medium', 'large']

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    required=True)
    parser.add_argument('-model',
                    help='name of model;',
                    type=str,
                    required=True)
    parser.add_argument('--num-rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=-1)
    parser.add_argument('--eval-every',
                    help='evaluate every ____ rounds;',
                    type=int,
                    default=-1)
    parser.add_argument('--clients-per-round',
                    help='number of clients trained per round;',
                    type=int,
                    default=-1)
    parser.add_argument('--batch-size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=10)
    parser.add_argument('--seed',
                    help='seed for random client sampling and batch splitting',
                    type=int,
                    default=0)
    parser.add_argument('--metrics-name', 
                    help='name for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)
    parser.add_argument('--metrics-dir', 
                    help='dir for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)
    parser.add_argument('--use-val-set', 
                    help='use validation set;', 
                    action='store_true')
    parser.add_argument("--agg-fn",
                    help='aggregation funciton',
                    type=str,
                    default="none-uniform",
                    required=False
    )

    # wandb initilization
    parser.add_argument(
        "--enable-wandb",
        type=str2bool,
        default=True
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="quantity",
        required=False
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        default="",
        required=False
    )

    # Minibatch doesn't support num_epochs, so make them mutually exclusive
    epoch_capability_group = parser.add_mutually_exclusive_group()
    epoch_capability_group.add_argument('--minibatch',
                    help='None for FedAvg, else fraction;',
                    type=float,
                    default=None)
    epoch_capability_group.add_argument('--num-epochs',
                    help='number of epochs when clients train on data;',
                    type=int,
                    default=1)

    parser.add_argument('-t',
                    help='simulation time: small, medium, or large;',
                    type=str,
                    choices=SIM_TIMES,
                    default='large')
    parser.add_argument('-lr',
                    help='learning rate for local optimizers;',
                    type=float,
                    default=-1,
                    required=False)

    # robust aggregation hyper-param
    parser.add_argument(
        "--trimmed-mean-beta",
        help="Number of params to filter in trimmed mean",
        type=int,
        default=1,
        required=False
    )
    parser.add_argument(
        "--krum-mal-num",
        help="Number of estimated number of malicious clients in krum",
        type=int,
        default=1,
        required=False
    )
    parser.add_argument(
        "--multi-krum-num",
        help="Number of cients selected in multi-krum",
        type=int,
        default=1,
        required=False
    )

    return parser.parse_args()
