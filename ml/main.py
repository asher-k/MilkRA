"""
python main.py --type processed --seed 1 --num_states 60 --save --experiment EXP --model MOD --name NAM --verbose
"""
import os
import logging
import sys
import numpy.random as nprand
import experiments as exp

from argparse import ArgumentParser, BooleanOptionalAction
from data import format_name, load


def define_arguments():
    """
    Establishes default model & training/inference parameters.

    :return: Default command-line arguments
    """
    a = ArgumentParser()
    a.add_argument('--seed', default=1, type=int,
                   help='Initial super seed of random for generating random seeds')
    a.add_argument('--num_states', default=1, type=int,
                   help='Number of random states to compute model performances at')
    a.add_argument('--dir', default='../data/processed', type=str,
                   help='Path to data folders')
    a.add_argument('--type', default='processed', type=str, choices=['processed', 'raw'],
                   help='Observations contain a raw and processed csv file')
    a.add_argument('--logs_dir', default='../logs/', type=str,
                   help='Logging directory')
    a.add_argument('--name', default='run', type=str,
                   help='Experiment name appended to files')
    a.add_argument('--save', default=False, action=BooleanOptionalAction,
                   help='Save performance statistics to a CSV in the logging directory')
    a.add_argument('--only_acc', default=False, action=BooleanOptionalAction,
                   help='Only save direct model outputs')
    a.add_argument('--verbose', default=False, action=BooleanOptionalAction,
                   help='Prints performance statistics to console & log and enables export of experiment figures')
    a.add_argument('--importance', default=False, action=BooleanOptionalAction,
                   help='Log feature importances from valid baseline models')
    a.add_argument('--experiment', default='classify:baseline', type=str, choices=["classify:baseline", "classify:dl",
                                                                                   "classify:ts", "cluster"],
                   help='Experiment to perform; assumes the model chosen is relevant for it')
    a.add_argument('--model', default='logreg', type=str,
                   help='ML baseline to obtain results on; can be \'all\' to sequentially run all baselines.')
    a.add_argument('--load_only', default=None, type=int,
                   help='Only load droplet at the given time step')
    a.add_argument('--load_at', nargs="+", type=int,
                   help='Appends droplet data at multiple time steps for dimensionality reduction')
    a.add_argument('--load_ranges', nargs="+", type=str,
                   help='String-formatted ranges representing indicies of steps to use in ML baselines')
    a.add_argument('--features_at', nargs="+", type=int,
                   help='Columns for dimensionality reduction; non-indexed columns are dropped prior to training')
    a.add_argument('--features_selection', default="none", choices=["none", "pca", "top"], type=str,
                   help='Perform PCA on the raw/processed datasets')
    a.add_argument('--centre_avg', default=False, action=BooleanOptionalAction,
                   help='Average centre 3 observations')
    a.add_argument('--normalize', default="max", choices=["max", "const", "none"], type=str,
                   help='Type of normalization to apply to droplet heights. Max normalizes according to the highest '
                        'observed droplet height, const according to a constant parameter')
    a = a.parse_args()
    return a


def preconditions(a):
    """
    Script precondition checks; produces log warnings and potentially stops execution. Ensure command-line arguments are
    within expected bounds.

    :param a: ArgParser object
    """
    if a.centre_avg and a.load_only is not None:
        raise ValueError("Simultaneous centre_avg and load_only is unsupported; please run with only one argument.")
    if a.load_only is not None and a.load_at is not None:
        logging.warning("Received arguments for load_only and load_at; ignoring --load_at {la}".format(la=a.load_at))
    if a.features_at is not None and a.features_selection != "none":
        logging.warning(
            "Received arguments for features_at and features_selection; selected features will rely on the subset")
    if not a.save and not a.verbose:
        logging.warning("Saving and Verbosity are both disabled! Only partial results are obtainable through log files")


# Delegate mode to correct model, using the provided arguments.
if __name__ == '__main__':
    args = define_arguments()
    preconditions(args)

    # logging init
    logs_dir = "{ld}{td}/{ed}/".format(ld=args.logs_dir, td=args.type, ed=args.name)
    logs_name = format_name(args, d=logs_dir, ext=".txt")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logging.basicConfig(filename=logs_name,
                        format="%(asctime)s: %(message)s",
                        filemode="w")
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True
    if args.verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # also print to console
    logging.info("Starting Preprocessing...")

    # load & reformat datasets
    nprand.seed(args.seed)
    data, labels = load(args.dir, args.type,
                        centre_avg=args.centre_avg,
                        only=args.load_only,
                        at=args.load_at,
                        ranges=args.load_ranges,
                        features=args.features_at,
                        normalize=args.normalize,
                        ts=any([True if d in args.experiment else False for d in ["ts", "dl"]])
                        )

    # delegate to experiment script
    logging.info(f"Delegating: {args.experiment}, {args.model}")
    if args.experiment == "classify:baseline":
        exp.classify_baselines(args, data, labels, logs_dir)
    elif args.experiment == "classify:dl":
        exp.classify_dl(args, data, labels)
    elif args.experiment == "classify:ts":
        exp.classify_ts(args, data, labels, logs_dir)
    elif args.experiment == "cluster":
        exp.clustering(args, data, labels)
    else:
        raise ValueError("Unable to delegate experiment type {exp}".format(exp=args.experiment))
