"""
python main.py --type processed --seed 1 --num_states 30 --model
"""
import os
import logging
from argparse import ArgumentParser
from methods import models
from sklearn.model_selection import train_test_split
import pandas as pd
import random


def define_arguments():
    """
    Establishes default model & training/inference parameters.

    :return: default arguments
    """
    a = ArgumentParser()
    a.add_argument('--model', default='logreg', type=str, help='ML classification model')
    a.add_argument('--dir', default='../data/processed', type=str, help='Path to data folders')
    a.add_argument('--type', default='processed', type=str, help='Observations contain a raw and processed csv file')
    a.add_argument('--seed', default=1, type=int, help='Initial seed of random for generating random seeds')
    a.add_argument('--num_states', default=1, type=int, help='Number of random states to compute model performances at')

    a.add_argument('--logs_dir', default='../logs/', type=str, help='Logging directory')
    a = a.parse_args()
    return a


def load(data_dir, data_type):
    """
    Loads preprocessed samples of droplet sequences and converts them into
    :return:
    """
    d = []
    l = []
    classes = os.listdir(data_dir)
    for c in classes:
        class_dir = data_dir + "/" + c
        insts = []
        for f in os.listdir(class_dir+"/"):  # individually load each .csv file
            file_dir = class_dir+"/"+f+"/"
            files = os.listdir(file_dir)
            file = list(filter(lambda fl: data_type+".csv" in fl, files))
            file = file[0]
            insts.append(pd.read_csv(file_dir+file))
        insts = [i.iloc[5:200:5, 2:] for i in insts]  # only every 5th value in first 200 timesteps, starting from 5

        to_avg = ["dl_height_midpoint", "2l_to_2r", "1l_to_1r"]
        for i in insts:  # get mean over centre 3 observations
            i["midpoints_mean"] = i[to_avg].mean(axis=1)
        insts = [i.drop(to_avg, axis=1) for i in insts]  # then drop original observations
        insts = [i.to_numpy().flatten() for i in insts]  # flatten instances
        d += insts
        l += [c] * len(insts)
    d = pd.DataFrame(d)  # then convert back into dataframe
    d = d.fillna(0)  # 0-imputation
    return d, l


# Delegate mode to correct model, using the provided arguments.
if __name__ == '__main__':
    args = define_arguments()
    logs_dir = args.logs_dir
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    logging.basicConfig(filename=logs_dir+"{name}.txt".format(name="run"), level=logging.DEBUG,
                        format="%(asctime)s: %(message)s", filemode="w")

    # load & reformat
    random.seed(args.seed)
    data, labels = load(args.dir, args.type)
    train_d, test_d, train_l, test_l = train_test_split(data, labels, test_size=0.3, stratify=labels)

    # execute models
    for model in models[args.model]:
        r = []
        for state in random.sample(range(0, 100000), args.num_states):
            r.append(model(train_d, train_l, test_d, test_l, random_state=state))

        results = pd.DataFrame(r)
        # results = results.mean(axis=0)
        print(results)
