"""
python main.py --type processed --seed 1 --num_states 60 --normalize --save --importance --model
"""
import os
import logging
import sys
import pandas as pd
import numpy as np
import numpy.random as nprand
from argparse import ArgumentParser, BooleanOptionalAction
from baseline import Baselines, Clustering
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.model_selection import train_test_split
from data import format_name, load, _col_order, run_pca
from plots import samplewise_misclassification_rates, confusion_matrix


def define_arguments():
    """
    Establishes default model & training/inference parameters.

    :return: default arguments
    """
    a = ArgumentParser()
    a.add_argument(
        '--seed', default=1, type=int,
        help='Initial super seed of random for generating random seeds'
    )
    a.add_argument(
        '--num_states', default=1, type=int,
        help='Number of random states to compute model performances at'
    )
    a.add_argument(
        '--dir', default='../data/processed', type=str,
        help='Path to data folders'
    )
    a.add_argument(
        '--type', default='processed', type=str, choices=['processed', 'raw'],
        help='Observations contain a raw and processed csv file'
    )
    a.add_argument(
        '--logs_dir', default='../logs/', type=str,
        help='Logging directory'
    )
    a.add_argument(
        '--name', default='run', type=str,
        help='Experiment name appended to files'
    )
    a.add_argument(
        '--save', default=False, action=BooleanOptionalAction,
        help='Save performance statistics to a CSV in the logging directory'
    )
    a.add_argument(
        '--only_acc', default=False, action=BooleanOptionalAction,
        help='Only save direct model outputs'
    )
    a.add_argument(
        '--verbose', default=False, action=BooleanOptionalAction,
        help='Print performance statistics to console in addition to the logging file, and enable DT figure export'
    )
    a.add_argument(
        '--importance', default=False, action=BooleanOptionalAction,
        help='Log feature importances from valid models'
    )
    a.add_argument(
        '--model', default='logreg', type=str,
        help='ML baseline to obtain results on; can be \'all\' to sequentially run all baselines.'
    )
    a.add_argument(
        '--load_only', default=None, type=int,
        help='Only load droplet at the given time step'
    )
    a.add_argument(
        '--load_at', nargs="+", type=int,
        help='Appends droplet data at multiple time steps for dimensionality reduction'
    )
    a.add_argument(
        '--load_ranges', nargs="+", type=str,
        help='String-formatted ranges representing indicies of steps to use in ML baselines'
    )
    a.add_argument(
        '--features_at', nargs="+", type=int,
        help='Columns to use for dimensionality reduction; non-indexed columns are dropped prior to training'
    )
    a.add_argument(
        '--features_selection', default="none", choices=["none", "pca", "top"], type=str,
        help='Perform PCA on the raw/processed datasets'
    )
    a.add_argument(
        '--centre_avg', default=False, action=BooleanOptionalAction,
        help='Average centre 3 observations'
    )
    a.add_argument(
        '--normalize', default="max", choices=["max", "const", "none"], type=str,
        help='Type of normalization to apply to droplet heights. Max normalizes according to the highest observed '
             'droplet height, const according to a constant parameter'
    )
    a = a.parse_args()
    return a


def preconds(a):
    """
    Script precondition checks
    """
    if a.centre_avg and a.load_only is not None:
        raise ValueError("Simultaneous centre_avg and load_only is unsupported; please run with only one argument.")
    if a.load_only is not None and a.load_at is not None:
        logging.warning("Received arguments for load_only and load_at; ignoring --load_at {la}".format(la=a.load_at))
    if a.features_at is not None and a.features_selection != "none":
        logging.warning(
            "Received arguments for features_at and features_selection; selected features will rely on the subset")
    if a.model not in list(baselines.m.keys()) + list(clusters.m.keys()) + ["CNN"]:
        raise ValueError("Unable to delegate model type {model}".format(model=a.model))
    if not a.save and not a.verbose:
        logging.warning("Saving and Verbosity are both disabled! Only partial results are obtainable through log files")


# Delegate mode to correct model, using the provided arguments.
if __name__ == '__main__':
    baselines = Baselines()
    clusters = Clustering()

    # preconditions & warnings
    args = define_arguments()
    preconds(args)

    # logging init
    logs_dir = "{ld}{td}/{ed}/".format(ld=args.logs_dir, td=args.type, ed=args.name)
    logs_name = format_name(args, d=logs_dir, ext=".txt")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logging.basicConfig(filename=logs_name,
                        level=logging.DEBUG,
                        format="%(asctime)s: %(message)s",
                        filemode="w")
    if args.verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # also print to console
    logging.info("run")

    # load & reformat datasets
    nprand.seed(args.seed)
    data, labels = load(args.dir, args.type,
                        centre_avg=args.centre_avg,
                        only=args.load_only,
                        at=args.load_at,
                        ranges=args.load_ranges,
                        features=args.features_at,
                        normalize=args.normalize)
    for row in data.index:  # inits samplewise misclassification counts
        baselines.preddict[row] = (0., 0)

    # Deep Learning Experiments
    if args.model == "CNN":
        exit()

    # execute clustering experiments
    if args.model not in baselines.m.keys():
        if args.features_selection == "pca":  # PCA can be deterministic under randomizer solver; may occur in BD
            data = run_pca()

        clusters.data(data, labels)
        model = clusters.m[args.model][0]
        model = model()
        clusters.dendrogram(model)
        exit()

    # execute classification experiments
    for model in baselines.m[args.model]:
        # can infer model name from baseline classifier keys
        if args.model == "all":
            model_name = list(baselines.m.keys())[baselines.m[args.model].index(model)]
        else:
            model_name = list(baselines.m.keys())[list(baselines.m.values()).index([model])]

        state_data = data.copy()
        if args.features_selection == "top":  # since Percentile is deterministic no need to run @ each seed
            selector = SelectPercentile(score_func=mutual_info_classif, percentile=20)
            selector.fit(state_data, labels)
            state_data = pd.DataFrame(selector.transform(state_data))

        # obtain model results over n seeds
        r, cm, p, dt_top = [], [], [], []  # results, confusion matricies, feature importance, decision tree splits
        for state in nprand.randint(0, 99999, size=args.num_states):
            if args.features_selection == "pca":  # PCA can be deterministic under randomizer solver; may occur in BD
                state_data = run_pca(state_data, labels, state)

            train_d, test_d, train_l, test_l = train_test_split(state_data, labels, test_size=0.3, stratify=labels)
            baselines.data(train_d, train_l, test_d, test_l)
            result, c, _imp, _split = model(random_state=state,
                                            verbose=args.verbose,
                                            importance=args.importance,
                                            feature_names=_col_order(args.type),
                                            only_acc=args.only_acc)
            for tr, re in zip([r, cm, p, dt_top], [result, c, _imp, _split]):
                tr.append(re)

        # aggregate results over all seeds & log it!
        results = pd.DataFrame(r)
        stddevs = results.std()  # light post-run averaging
        results = results.mean(axis=0).round(decimals=3)
        for col, val in stddevs.items():  # reformatting stddev column names
            results[col+"+-"] = val
        results = results.to_frame().transpose()
        results = results.reindex(sorted(results.columns), axis=1)
        logging.info(msg="\nPerformance Statistics: {mod}\n{res}\n".format(mod=model_name, res=str(results)))
        save_dir = "{log}{mod}/".format(log=logs_dir, mod=model_name)

        # save performance results to csv file(s) with any produced plots
        if args.save:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            csv_name = format_name(args, save_dir, ".csv")
            results.to_csv(csv_name)

            if args.verbose:  # aggregate confusion matrix & misclassification rate figures & save to output directory
                samplewise_misclassification_rates(baselines, labels, args, save_dir, model_name)
                confusion_matrix(cm, labels, args, save_dir, model_name)

            # format & save feature importances into a logging subdirectory
            if args.importance and not args.only_acc and model_name in ["logreg", "dt"]:
                importance_dir = "{sd}importance/".format(sd=save_dir)
                if not os.path.exists(importance_dir):
                    os.mkdir(importance_dir)

                priority = pd.DataFrame(p, columns=_col_order(args.type))
                priority = priority.abs()
                cvs = np.cov(priority, rowvar=False)
                imp_name = format_name(args, importance_dir, ".npy")
                np.save(imp_name, cvs)

                priority = priority.mean(axis=0).round(decimals=3).to_frame().transpose()
                priority = priority.to_frame().transpose()
                imp_name = format_name(args, importance_dir, ".csv")
                priority.to_csv(imp_name)

                # Also save decision tree splits for analysis
                if model_name == "dt":
                    importance_splits_dir = "{id}splits/".format(id=importance_dir)
                    if not os.path.exists(importance_splits_dir):
                        os.mkdir(importance_splits_dir)
                    dt_splits = pd.DataFrame(dt_top, columns=["top_split"])
                    imp_name = format_name(args, importance_splits_dir, ".csv")
                    dt_splits.to_csv(imp_name)
