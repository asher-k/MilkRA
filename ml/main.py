"""
python main.py --type processed --seed 1 --num_states 60 --normalize --save --model
"""
import os
import logging
import sys
import pandas as pd
import numpy.random as nprand
from argparse import ArgumentParser, BooleanOptionalAction
from baseline import models
from sklearn.model_selection import train_test_split


def define_arguments():
    """
    Establishes default model & training/inference parameters.

    :return: default arguments
    """
    a = ArgumentParser()
    a.add_argument('--seed', default=1, type=int, help='Initial seed of random for generating random seeds')
    a.add_argument('--num_states', default=1, type=int, help='Number of random states to compute model performances at')

    a.add_argument('--dir', default='../data/processed', type=str, help='Path to data folders')
    a.add_argument('--type', default='processed', type=str, choices=['processed', 'raw'],
                   help='Observations contain a raw and processed csv file')
    a.add_argument('--logs_dir', default='../logs/', type=str, help='Logging directory')
    a.add_argument('--name', default='run', type=str, help='Logging directory')
    a.add_argument('--save', default=False, action=BooleanOptionalAction, help='Save performance statistics to a CSV '
                                                                               'in the logging directory')
    a.add_argument('--verbose', default=False, action=BooleanOptionalAction, help='When true print performance '
                                                                                  'statistics to console as well as '
                                                                                  'logging file')

    a.add_argument('--model', default='logreg', type=str, help='ML classification model')
    a.add_argument('--load_only', default=None, type=int, help='Only load droplet sequences at the provided timestep')
    a.add_argument('--centre_avg', default=False, action=BooleanOptionalAction, help='Average centre 3 observations')
    a.add_argument('--normalize', default=False, action=BooleanOptionalAction, help='Normalize droplet heights to the '
                                                                                    'First midpoint observation in '
                                                                                    'the sequence')
    a = a.parse_args()
    return a


def _col_order(data_type):
    """
    Returns constants representing column orders defined externally by the data

    :param data_type: type of data being used: either raw or processed
    :return:
    """
    if data_type == "processed":
        return ['edge_4_r_to_edge_4_l', 'edge_3_r_to_edge_3_l', '11l_to_11r', 'edge_2_r_to_edge_2_l',
                'edge_1_r_to_edge_1_l', '10l_to_10r', '9l_to_9r', '8l_to_8r', '7l_to_7r', '6l_to_6r', '5l_to_5r',
                '4l_to_4r', '3l_to_3r', '2l_to_2r', '1l_to_1r', 'dl_height_midpoint']
    return ['edge_4_l', 'edge_3_l', '11l', 'edge_2_l', 'edge_1_l', '10l', '9l', '8l', '7l', '6l', '5l', '4l', '3l',
            '2l', '1l', 'dl_height_midpoint', '1r', '2r', '3r', '4r', '5r', '6r', '7r', '8r', '9r', '10r', 'edge_1_r',
            'edge_2_r', '11r', 'edge_3_r', 'edge_4_r']


def load(data_dir, data_type, **kwargs):
    """
    Loads preprocessed samples of droplet sequences, including
    :return:
    """
    d = []
    labs = []
    norm_consts = []
    classes = os.listdir(data_dir)
    for c in classes:
        class_dir = data_dir + "/" + c
        seqs = []
        for f in os.listdir(class_dir+"/"):  # individually load each .csv file
            file_dir = class_dir+"/"+f+"/"
            files = os.listdir(file_dir)
            file = list(filter(lambda fl: data_type+".csv" in fl, files))
            file = pd.read_csv(file_dir+file[0])
            seqs.append(file)
        index_cols = 2 if "processed" in data_type else 4  # depending on raw vs processed, different num of ref. cols.
        [norm_consts.append(s.iloc[0, index_cols]) for s in seqs]  # track normalization constant for each droplet seq.
        seqs = [s[_col_order(data_type)] for s in seqs]  # reshape column orders
        seqs = [i.iloc[:900, :] for i in seqs] if kwargs['at'] is None \
            else [i.iloc[kwargs['at'], :] for i in seqs]

        if kwargs['centre_avg']:  # get mean over centre 3 observations
            to_avg = ["dl_height_midpoint", "2l_to_2r", "1l_to_1r"]
            for i in seqs:  #
                i["midpoints_mean"] = i[to_avg].mean(axis=1)
            seqs = [i.drop(to_avg, axis=1) for i in seqs]  # then drop original observations

        seqs = [i.to_numpy().flatten() for i in seqs]  # flatten instances
        d += seqs
        labs += [c] * len(seqs)
    d = pd.DataFrame(d)  # then convert back into dataframe
    d = d.fillna(0)  # 0-imputation
    if kwargs['normalize']:
        d = d.div(norm_consts, axis=0)
    return d, labs


# Delegate mode to correct model, using the provided arguments.
if __name__ == '__main__':
    args = define_arguments()
    if args.centre_avg and args.load_only is not None:
        raise ValueError("Simultaneous centre_avg and load_only is unsupported; please run with only one argument.")
    if args.model not in models.keys():
        raise ValueError("Unknown model type {model}".format(model=args.model))
    if not args.save and not args.verbose:
        logging.warning("Saving and Verbosity are both disabled! Only partial results are obtainable through log files")

    logs_dir = args.logs_dir
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    formatted_name = "{name}_{model}_{type}{norm}{avg}{only}"\
        .format(name=args.name, model=args.model, type=args.type, norm="_norm" if args.normalize else "",
                avg="_mpmean"if args.centre_avg else "", only="_"+str(args.load_only) if str(args.load_only) is not None else "")
    logging.basicConfig(filename=logs_dir+"{name}.txt".format(name=formatted_name), level=logging.DEBUG,
                        format="%(asctime)s: %(message)s", filemode="w")
    if args.verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # also print to console

    # load & reformat
    nprand.seed(args.seed)
    data, labels = load(args.dir, args.type, centre_avg=args.centre_avg, at=args.load_only, normalize=args.normalize)

    # execute models
    for model in models[args.model]:
        if args.model == "all":  # can infer model name from model keys
            model_name = list(models.keys())[models[args.model].index(model)]
        else:
            model_name = list(models.keys())[list(models.values()).index([model])]

        r = []
        p = []
        for state in nprand.randint(0, 99999, size=args.num_states):
            train_d, test_d, train_l, test_l = train_test_split(data, labels, test_size=0.3, stratify=labels)
            result, imp = model(train_d, train_l, test_d, test_l, random_state=state)
            if imp is not None:
                p.append(imp)
            r.append(result)

        results = pd.DataFrame(r)
        stddevs = results.std()  # light post-run averaging
        results = results.mean(axis=0).round(decimals=3)
        for col, val in stddevs.iteritems():  # reformatting stddev column names
            results[col+"+-"] = val
        results = results.to_frame().transpose()
        results = results.reindex(sorted(results.columns), axis=1)

        logging.info(msg="\nPerformance Statistics\n"+str(results)+"\n")
        save_dir = "{log}/{mod}".format(log=logs_dir, mod=model_name)

        if args.save:  # save results to a csv file
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            results.to_csv("{save}/{name}{type}{norm}{avg}{only}.csv"
                           .format(name=args.name, save=save_dir, type=args.type, norm="_norm" if args.normalize else ""
                                   , avg="_mpmean" if args.centre_avg else "",
                                   only="_"+str(args.load_only) if str(args.load_only) is not None else ""))

        if args.save and p:  # format & save feature priorities into a subdirectory
            priority = pd.DataFrame(p, columns=_col_order(args.type))
            priority = priority.abs()
            priority = priority.mean(axis=0).round(decimals=3)
            priority = priority.to_frame().transpose()
            if not os.path.exists(save_dir+"/importance"):
                os.mkdir(save_dir+"/importance")
            priority.to_csv("{save}/importance/{name}{type}{norm}{avg}{only}.csv"
                            .format(name=args.name, save=save_dir, type=args.type, norm="_norm" if args.normalize else
                            "", avg="_mpmean" if args.centre_avg else "", only="_"+str(args.load_only) if
                            str(args.load_only) is not None else ""))


