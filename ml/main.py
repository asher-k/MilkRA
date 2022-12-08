"""
python main.py --type processed --seed 1 --num_states 30 --model
"""
import os
import logging
import sys
import random
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
    a.add_argument('--type', default='processed', type=str, help='Observations contain a raw and processed csv file')
    a.add_argument('--logs_dir', default='../logs/', type=str, help='Logging directory')

    a.add_argument('--model', default='logreg', type=str, help='ML classification model')
    a.add_argument('--load_only', default=None, type=int, help='Only load droplet sequences at the provided timestep')
    a.add_argument('--centre_avg', default=False, action=BooleanOptionalAction, help='Average centre 3 observations')
    a.add_argument('--normalize', default=False, action=BooleanOptionalAction, help='Normalize droplet heights to the '
                                                                                    'First midpoint observation in '
                                                                                    'the sequence')
    a = a.parse_args()
    return a


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
        seqs = [i.iloc[:900, index_cols:] for i in seqs] if kwargs['at'] is None \
            else [i.iloc[kwargs['at'], index_cols:] for i in seqs]

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
    logs_dir = args.logs_dir
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    logging.basicConfig(filename=logs_dir+"{name}.txt".format(name="run"), level=logging.DEBUG,
                        format="%(asctime)s: %(message)s", filemode="w")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # also print to console

    # load & reformat
    nprand.seed(args.seed)
    data, labels = load(args.dir, args.type, centre_avg=args.centre_avg, at=args.load_only, normalize=args.normalize)

    # execute models
    for model in models[args.model]:
        r = []
        for state in nprand.randint(0, 99999, size=args.num_states):
            train_d, test_d, train_l, test_l = train_test_split(data, labels, test_size=0.3, stratify=labels)
            result = model(train_d, train_l, test_d, test_l, random_state=state)
            r.append(result)

        results = pd.DataFrame(r)
        results = results.mean(axis=0).round(decimals=3)
        logging.info(msg="\nPerformance Statistics\n"+str(results))
