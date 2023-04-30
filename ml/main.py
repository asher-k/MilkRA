"""
python main.py --type processed --seed 1 --num_states 60 --save --experiment EXP --model MOD --name NAM --verbose
"""
import shutil

import os
import logging
import sys
import json
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
    # Experiment & Logging Arguments
    a.add_argument('--name', default='run', type=str,
                   help='Experiment name appended to files')
    a.add_argument('--seed', default=1, type=int,
                   help='Initial root seed of random for generating random seeds')
    a.add_argument('--verbose', default=False, action=BooleanOptionalAction,
                   help='Prints performance statistics to console & log and enables export of experiment figures')
    a.add_argument('--dir', default='../data/processed', type=str,
                   help='Path to data folders')
    a.add_argument('--out_dir', default='../output/', type=str,
                   help='Path to logs directory')
    a.add_argument('--save', default=False, action=BooleanOptionalAction,
                   help='Save performance statistics to a CSV in the logging directory; also saves PyTorch models')
    a.add_argument('--overwrite', default=False, action=BooleanOptionalAction,
                   help='Overwrite any files present in the current experiment folder')
    a.add_argument('--load', default=False, action=BooleanOptionalAction,
                   help='Loads pretrained PyTorch models from out_dir')

    # Model arguments
    a.add_argument('--experiment', default='classify:baseline', type=str, choices=["classify:baseline", "classify:dl",
                                                                                   "classify:ts", "classify:vit",
                                                                                   "cluster", "pca"],
                   help='Experiment to perform; assumes the model chosen is relevant for it')
    a.add_argument('--model', default='logreg', type=str,
                   help='ML baseline to obtain results on; can be \'all\' to sequentially run all baselines.')
    a.add_argument('--num_states', default=1, type=int,
                   help='Number of random states to compute model performances across')

    # Non-TS Baseline arguments
    a.add_argument('--only_acc', default=False, action=BooleanOptionalAction,
                   help='Disables the export of most figures except for ones concerned with accuracy/misclassification')
    a.add_argument('--importance', default=False, action=BooleanOptionalAction,
                   help='Export feature importance visualizations (when applicable with the selected model)')

    # PSO-PCA experiment arguments
    a.add_argument('--pso_prop', default=500, type=int,
                   help='Proportion of particles initialized compared to the possible number; the denominator of '
                        '(possible_points **2) / pso_prop')
    a.add_argument('--pso_initsize', default=100, type=int,
                   help='(approx.) Number of features to randomly initialize in a particle.')
    a.add_argument('--pso_initscheme', default='stochastic', type=str, choices=["stochastic", "deterministic"],
                   help='If stochastic, particles are initialized by probability, leading to variance between the number'
                        'of chosen timesteps. If deterministic, all particles will have pso_initsize timesteps intially.')
    a.add_argument('--pso_iters', default=50, type=int,
                   help='Number of training iterations to perform')

    # PyTorch Deep Learning arguments
    a.add_argument('--pyt_lr', default=1e-3, type=float,
                   help='Model Learning Rate')
    a.add_argument('--pyt_bs', default=6, type=int,
                   help='Batch size; smaller values preferred due to low sample count')
    a.add_argument('--pyt_epochs', default=50, type=int,
                   help='Number of training epochs')
    a.add_argument('--pyt_data_split', default=(0.667, 0.333), type=tuple,
                   help='Tuple of (train split, test split) floats. Should sum to 1.0')
    a.add_argument('--vit_subdiv_size', default=4, type=int,
                   help='Subdivision size of our droplet samples for our ViT')
    a.add_argument('--vit_dims', default=16, type=int,
                   help='Hidden dimensions of our ViT embedding')
    a.add_argument('--vit_heads', default=4, type=int,
                   help='Number of Attention Heads within our MHSA module')
    a.add_argument('--vit_blocks', default=1, type=int,
                   help='Number of transformer blocks within our ViT')

    # Data/preprocessing arguments
    a.add_argument('--type', default='processed', type=str, choices=['processed', 'raw'],
                   help='Samples contain a raw and processed csv file; this designates which one is used by our models')
    a.add_argument('--load_at', nargs="+", type=int,
                   help='Concatenates flattened droplet data at the specified time steps')
    a.add_argument('--load_ranges', nargs="+", type=str,
                   help='String-formatted ranges representing indicies of steps to use in ML baselines')
    a.add_argument('--features_at', nargs="+", type=int,
                   help='Columns for dimensionality reduction; non-indexed columns are dropped prior to training')
    a.add_argument('--features_selection', default="none", choices=["none", "pca", "umap", "top"], type=str,
                   help='Perform PCA on the raw/processed dataset')
    a.add_argument('--centre_avg', default=False, action=BooleanOptionalAction,
                   help='Average centre 3 observations; this could provide a boost in performance through dim. red.')
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
    if a.centre_avg and a.features_at is not None:
        raise ValueError("Simultaneous centre_avg and load_only is unsupported; please run with only one argument.")
    if a.features_at is not None and a.features_selection != "none":
        logging.warning(
            "Received arguments for features_at and features_selection; selected features will rely on the subset")
    if not a.save and not a.verbose:
        logging.warning("Saving and Verbosity are both disabled! Full results may not be available.")
    if a.save and a.num_states > 1 and any([t in a.experiment for t in ['dl', 'vit']]):
        logging.warning("Saving is enabled for multiple PyTorch models! All models will save at the cost of disk space")
    if a.save and a.overwrite:
        logging.critical(f"OVERWRITING output folder {a.name}")


# Delegate mode to correct model, using the provided arguments.
if __name__ == '__main__':
    args = define_arguments()

    out_dir = "{od}experiments/{e}/".format(od=args.out_dir, e=args.name)
    logs_name = f"{out_dir}logs_{args.name}.txt"

    if not args.load:  # directory initialization
        for d in [f"{out_dir}figs/", f"{out_dir}models/", f"{out_dir}results/"]:
            if not os.path.exists(d):
                os.makedirs(d)
            elif args.overwrite:
                shutil.rmtree(d)
                os.makedirs(d)
            else:
                raise ValueError(f"Files already exist for experiment {args.name}! "
                                 f"Please re-run under a different name or enable overwriting via --overwrite.")
        with open(f'{out_dir}args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        preconditions(args)  # check command-line arguments are valid
    else:  # Load from a provided model checkpoint
        args_dict = json.load(open(f"{out_dir}args.txt"))
        for key in args_dict.keys():
            if key != "load":  # Don't want to overwrite intended script mode
                args.__dict__[key] = args_dict[key]
    fh = logging.FileHandler(logs_name, mode="w")  # logging initialization
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger("EXPR_LOG")
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logger.info("Starting Preprocessing...")

    # load & reformat datasets
    nprand.seed(args.seed)
    data, labels = load(args.dir, args.type,
                        centre_avg=args.centre_avg,
                        at=args.load_at,
                        ranges=args.load_ranges,
                        features=args.features_at,
                        normalize=args.normalize,
                        ts=any([True if d in args.experiment else False for d in ["ts", "dl", "vit", "pca"]])  # is TS?
                        )

    # delegate to experiment script
    logger.info(f"Delegating: {args.experiment}, {args.model}")
    if args.experiment == "classify:baseline":
        exp.classify_baselines(args, data, labels, out_dir, logger)
    elif args.experiment == "classify:ts":
        exp.classify_ts(args, data, labels, out_dir, logger)
    elif args.experiment == "cluster":
        exp.clustering(args, data, labels, out_dir, logger)
    elif args.experiment == "pca":
        exp.pso(args, data, labels, out_dir, logger)
    elif args.experiment == "classify:dl":
        exp.classify_dl(args, data, labels, out_dir, logger)
    elif args.experiment == "classify:vit":
        exp.classify_vit(args, data, labels, out_dir, logger)
    else:
        raise ValueError("Unable to delegate experiment type {exp}".format(exp=args.experiment))
