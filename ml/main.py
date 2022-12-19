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
from baseline import Baselines
from sklearn.model_selection import train_test_split


def _col_order(data_type):
    """
    Returns constants representing column orders defined externally by the data

    :param data_type: type of data being used: either raw or processed
    :return: column orders for the corresponding dataset
    """
    if data_type == "processed":
        return ['edge_4_r_to_edge_4_l', 'edge_3_r_to_edge_3_l', '11l_to_11r', 'edge_2_r_to_edge_2_l',
                'edge_1_r_to_edge_1_l', '10l_to_10r', '9l_to_9r', '8l_to_8r', '7l_to_7r', '6l_to_6r', '5l_to_5r',
                '4l_to_4r', '3l_to_3r', '2l_to_2r', '1l_to_1r', 'dl_height_midpoint']
    return ['edge_4_l', 'edge_3_l', '11l', 'edge_2_l', 'edge_1_l', '10l', '9l', '8l', '7l', '6l', '5l', '4l', '3l',
            '2l', '1l', 'dl_height_midpoint', '1r', '2r', '3r', '4r', '5r', '6r', '7r', '8r', '9r', '10r', 'edge_1_r',
            'edge_2_r', '11r', 'edge_3_r', 'edge_4_r']


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
        help='Only load droplet sequences at the provided timestep'
    )
    a.add_argument(
        '--load_at', nargs="+", type=int,
        help='Appends droplet data at these times for dimensionality red.'
    )
    a.add_argument(
        '--centre_avg', default=False, action=BooleanOptionalAction,
        help='Average centre 3 observations'
    )
    a.add_argument(
        '--normalize', default=False, action=BooleanOptionalAction,
        help='Normalize droplet heights to the First midpoint observation in the sequence'
    )
    a = a.parse_args()
    return a


def format_name(arg, d=None, ext=None):
    """
    Produces a formatted file name

    :param arg: command-line arguments used to construct file name
    :param d: export directory prepended to name
    :param ext: file extension appended to name
    :return:
    """
    fname = "{save_dir}{name}.{model}.{type}.{norm}.{avg}{only}{ext}"
    fname = fname.format(
        save_dir=d,
        name=arg.name,
        model=arg.model,
        type=arg.type,
        norm="norm" if arg.normalize else "",
        avg="mpmean." if arg.centre_avg else "",
        only=""+str(arg.load_only) if str(arg.load_only) is not None else "",
        ext=ext
    )
    return fname


def load(data_dir, data_type, **kwargs):
    """
    Loads preprocessed samples of droplet sequences, including
    :return:
    """
    x, y, norm_consts = [], [], []
    classes = os.listdir(data_dir)
    for c in classes:
        class_dir = "{d}/{c}".format(d=data_dir, c=c)
        seqs, index_cols = [], []
        for f in os.listdir(class_dir+"/"):  # individually load each .csv file
            file_dir = "{c}/{f}/".format(c=class_dir, f=f)
            files = os.listdir(file_dir)
            file = list(filter(lambda fl: data_type+".csv" in fl, files))
            file = pd.read_csv(f"{file_dir}{file[0]}")
            index_cols.append(file.columns.get_loc("dl_height_midpoint"))
            seqs.append(file)
        [norm_consts.append(s.iloc[0, i]) for s, i in zip(seqs, index_cols)]
        seqs = [s[_col_order(data_type)] for s in seqs]  # reshape column orders
        seqs = [i.iloc[:900, :] for i in seqs] if kwargs['only'] is None else [i.iloc[kwargs['only'], :] for i in seqs]
        seqs = seqs if kwargs['at'] is None else [i.iloc[kwargs['at'], :] for i in seqs]

        if kwargs['centre_avg']:  # get mean over centre 3 observations
            to_avg = ["dl_height_midpoint", "2l_to_2r", "1l_to_1r"]
            for i in seqs:  #
                i["midpoints_mean"] = i[to_avg].mean(axis=1)
            seqs = [i.drop(to_avg, axis=1) for i in seqs]  # then drop original observations
        seqs = [i.to_numpy().flatten() for i in seqs]  # flatten instances
        x += seqs
        y += [c] * len(seqs)
    x = pd.DataFrame(x)  # then convert back into dataframe
    x = x.fillna(0)  # 0-imputation
    if kwargs['normalize']:
        x = x.div(norm_consts, axis=0)
    return x, y


# Delegate mode to correct model, using the provided arguments.
if __name__ == '__main__':
    baselines = Baselines()

    # preconditions & warnings
    args = define_arguments()
    if args.centre_avg and args.load_only is not None:
        raise ValueError("Simultaneous centre_avg and load_only is unsupported; please run with only one argument.")
    if args.load_only is not None and args.load_at is not None:
        raise ValueError("Simultaneous load_at and load_only is unsupported; please run with only one argument.")
    if args.model not in baselines.m.keys():
        raise ValueError("Unknown model type {model}".format(model=args.model))
    if not args.save and not args.verbose:
        logging.warning("Saving and Verbosity are both disabled! Only partial results are obtainable through log files")

    # logging initialization
    logs_dir = "{ld}{td}/".format(ld=args.logs_dir, td=args.type)
    logs_name = format_name(args, d=logs_dir, ext=".txt")
    if not os.path.exists(args.logs_dir):
        os.mkdir(args.logs_dir)
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    logging.basicConfig(filename=logs_name,
                        level=logging.DEBUG,
                        format="%(asctime)s: %(message)s",
                        filemode="w")
    if args.verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # also print to console

    # load & reformat datasets
    nprand.seed(args.seed)
    data, labels = load(args.dir, args.type,
                        centre_avg=args.centre_avg,
                        only=args.load_only,
                        at=args.load_at,
                        normalize=args.normalize)

    # execute baseline experiments
    for model in baselines.m[args.model]:
        # can infer model name from model keys
        if args.model == "all":
            model_name = list(baselines.m.keys())[baselines.m[args.model].index(model)]
        else:
            model_name = list(baselines.m.keys())[list(baselines.m.values()).index([model])]

        # obtain model results over n seeds
        r, p, dt_top = [], [], []
        for state in nprand.randint(0, 99999, size=args.num_states):
            train_d, test_d, train_l, test_l = train_test_split(data, labels, test_size=0.3, stratify=labels)
            baselines.data(train_d, train_l, test_d, test_l)
            result, _imp, _split = model(random_state=state,
                                         verbose=args.verbose,
                                         importance=args.importance,
                                         feature_names=_col_order(args.type),
                                         only_acc=args.only_acc)
            r.append(result)
            p.append(_imp)
            dt_top.append(_split)

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
