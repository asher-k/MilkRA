"""
python main.py --type processed --seed 1 --num_states 60 --normalize --save --importance --model
"""
import os
import logging
import sys
import pandas as pd
import numpy as np
import numpy.random as nprand
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import sklearn.preprocessing as sklp
from argparse import ArgumentParser, BooleanOptionalAction
from baseline import Baselines, Clustering
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA

from data import format_name, load, _col_order


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


# Delegate mode to correct model, using the provided arguments.
if __name__ == '__main__':
    baselines = Baselines()
    cluster = Clustering()

    # preconditions & warnings
    args = define_arguments()
    if args.centre_avg and args.load_only is not None:
        raise ValueError("Simultaneous centre_avg and load_only is unsupported; please run with only one argument.")
    if args.load_only is not None and args.load_at is not None:
        logging.warning("Received arguments for load_only and load_at; ignoring --load_at {la}".format(la=args.load_at))
    if args.features_at is not None and args.features_selection != "none":
        logging.warning("Received arguments for features_at and features_selection; selected features will rely on the subset")
    if args.model not in list(baselines.m.keys()) + list(cluster.m.keys()):
        raise ValueError("Unknown model type {model}".format(model=args.model))
    if not args.save and not args.verbose:
        logging.warning("Saving and Verbosity are both disabled! Only partial results are obtainable through log files")

    # logging initialization
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
    for row in data.index:
        baselines.preddict[row] = (0., 0)

    # execute clustering experiments
    if args.model not in baselines.m.keys():
        if args.features_selection == "pca":  # PCA can be deterministic under randomizer solver; may occur in BD
            N = 5
            standardizer = sklp.StandardScaler().fit(data)
            dstd = standardizer.transform(data)
            pca = PCA(random_state=args.seed, n_components=N)  # >0.9 @ 2 PCs
            data = pd.DataFrame(pca.fit_transform(dstd, labels))
            logging.info(f"PCA explained variance: {pca.explained_variance_ratio_}")

        cluster.data(data, labels)
        model = cluster.m[args.model][0]
        model = model()
        cluster.dendrogram(model)
        plt.clf()
        exit()

    # execute classification experiments
    for model in baselines.m[args.model]:
        # can infer model name from model keys
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
                N = 5
                standardizer = sklp.StandardScaler().fit(state_data)
                dstd = standardizer.transform(state_data)
                pca = PCA(random_state=state, n_components=N)  # >0.9 @ 2 PCs
                state_data = pd.DataFrame(pca.fit_transform(dstd, labels))
                logging.info(f"PCA explained variance: {pca.explained_variance_ratio_}")

            train_d, test_d, train_l, test_l = train_test_split(state_data, labels, test_size=0.3, stratify=labels)
            baselines.data(train_d, train_l, test_d, test_l)
            result, c, _imp, _split = model(random_state=state,
                                            verbose=args.verbose,
                                            importance=args.importance,
                                            feature_names=_col_order(args.type),
                                            only_acc=args.only_acc)
            r.append(result)
            cm.append(c)
            p.append(_imp)
            dt_top.append(_split)

        # Plot sample-wise misclassification rates
        misc_rates = {str(k): round(v[0]/v[1], 3) for k, v in baselines.preddict.items()}
        misc_rates = pd.DataFrame(misc_rates.items(), columns=['id', 'misc']).sort_values('misc', ascending=False)
        misc_rates = misc_rates.loc[:][:22]
        misc_rates.reset_index(drop=True, inplace=True)
        lab_to_col = {"DBM": "mediumblue", "GTM": "forestgreen", "LBM": "dodgerblue", "LBP+": "goldenrod"}
        colors = [lab_to_col[labels[int(i)][:4].strip()] for i in misc_rates['id']]
        sns.set_context('paper')
        f, ax = plt.subplots(figsize=(6, 15))
        plt.xlim(0.0, 1.0)
        sns.set_color_codes('pastel')
        sns.barplot(x='misc', y='id', data=misc_rates, palette=colors, edgecolor='w')
        sns.despine(left=True, bottom=True)
        plt.xlabel("Misclassification Rate")
        plt.ylabel("Index")
        ax = plt.gca()
        for i in ax.containers:
            ax.bar_label(i, )
        patches = [mpatches.Patch(color=v, label=k) for k, v in lab_to_col.items()]
        ax.legend(handles=patches, loc='lower right')
        plt.show()

        # aggregate seed results & log
        results = pd.DataFrame(r)
        stddevs = results.std()  # light post-run averaging
        results = results.mean(axis=0).round(decimals=3)
        for col, val in stddevs.items():  # reformatting stddev column names
            results[col+"+-"] = val
        results = results.to_frame().transpose()
        results = results.reindex(sorted(results.columns), axis=1)
        logging.info(msg="\nPerformance Statistics: {mod}\n{res}\n".format(mod=model_name, res=str(results)))
        save_dir = "{log}{mod}/".format(log=logs_dir, mod=model_name)

        # save raw performance results to a csv file
        if args.save:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            csv_name = format_name(args, save_dir, ".csv")
            results.to_csv(csv_name)

            if args.verbose:  # aggregate confusion matrix & save to output directory
                plt.clf()
                cm = np.sum(cm, axis=0)
                cm = np.round(cm / np.sum(cm, axis=1), 3)
                cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted([l[0:4] for l in set(labels)]))
                cm_display.plot()
                fig_name = format_name(args, save_dir, f"_{model_name}.png")
                plt.savefig(fig_name)

        # format & save feature importances into a logging subdirectory
        if args.save and args.importance and not args.only_acc and model_name in ["logreg", "dt"]:
            importance_dir = "{sd}importance/".format(sd=save_dir)
            if not os.path.exists(importance_dir):
                os.mkdir(importance_dir)

            priority = pd.DataFrame(p, columns=_col_order(args.type))
            priority = priority.abs()
            cvs = np.cov(priority, rowvar=False)
            imp_name = format_name(args, importance_dir, ".npy")
            np.save(imp_name, cvs)

            priority = priority.mean(axis=0).round(decimals=3)
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
