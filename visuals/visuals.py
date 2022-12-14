"""
python visuals.py --data processed --metric Accuracy --importance --covariance
"""


import os
import re
from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

reshaped_order = ['edge_4_l', 'edge_3_l', '11l', 'edge_2_l', 'edge_1_l', '10l', '9l', '8l', '7l', '6l', '5l', '4l',
                  '3l', '2l', '1l', 'dl_height_midpoint', '1r', '2r', '3r', '4r', '5r', '6r', '7r', '8r', '9r', '10r',
                  'edge_1_r', 'edge_2_r', '11r', 'edge_3_r', 'edge_4_r']


def define_arguments():
    """
    Establishes default model & training/inference parameters.

    :return: default arguments
    """
    a = ArgumentParser()
    a.add_argument('--data', default="processed", type=str, choices=["processed", "raw"], help='Type of dataset')
    a.add_argument('--metric', default="Accuracy", type=str, help='Name of accuracy metric')
    a.add_argument('--importance', default=False, action=BooleanOptionalAction, help='Export feature import. figures')
    a.add_argument('--covariance', default=False, action=BooleanOptionalAction, help='Export feature covar. figures')
    a = a.parse_args()
    return a


def load_model_expr(path):
    """
    Loads experiment results from a provided directory

    :param path: path to experiment directory
    :return:
    """
    files = _load_and_sort(path, ".csv", sort=True)
    lines = list(reversed([pd.read_csv(path+file).iloc[:, 1:] for file in files]))
    lines = pd.concat(lines, ignore_index=True)

    lines["Timestep"] = list(reversed(_time_steps()))  # these are hardcoded; remove ASAP
    return lines


def load_model_expr_importances(path, processed=False):
    """
    Loads feature importance results from a provided directory

    :param path: path to experiment directory
    :param processed: when "processed" columns are duplicated to maintain symmetry when visualized
    :return:
    """
    files = _load_and_sort(path, ".csv")
    lines = list(reversed([pd.read_csv(path+file).iloc[:, 1:] for file in files]))
    lines = pd.concat(lines, ignore_index=True)
    if processed:
        lines = pd.concat([lines.iloc[:, :], lines.iloc[:, 14::-1]], axis=1)
        lines.columns = [str(i) for i in range(0, len(lines.columns))]
    return lines


def _load_droplet_example_shadow():
    """
    Loads an example droplet evaporation sequence as a shadow for feature importance analysis

    :return:
    """
    load_dir = "../data/processed/DBM 1000mA Repeats/221121AP/221121AP_raw.csv"
    shadow_data = pd.read_csv(load_dir)
    shadow_data = shadow_data.iloc[:, 4:]
    shadow_data = shadow_data[reshaped_order]
    return shadow_data


def _time_steps():
    """
    Hardcoded list of observed timestamps; TODO: should be inferred from valid files

    :return:
    """
    return [0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900]


def f1(data, pre, rec):
    """
    F1 score from precision & recall metrics
    """
    return 2 * ((data[pre]*data[rec])/(data[pre]+data[rec]))


def _load_and_sort(path, ext, sort=False):
    """
    Loads and sorts alphanumerically files from the provided directory.

    :param path: directory to load files
    :param ext: allowed file extension
    :param sort: True sorts file names alphanumerically
    :return:
    """
    files = os.listdir(path)
    files = [file for file in files if ext in file]
    if sort:
        files.sort(key=lambda f: int(re.split("[_.]", f)[-2]))
    return files


def accuracy_plot(mod, names, metric):
    """
    Plots the accuracy of the provided model in a line plot

    :return:
    """
    axis = plt.gca()
    axis.set_xlim([0, 900])
    axis.set_ylim([0.0, 1.0])
    for m, n in zip(mod, names):
        sns.lineplot(data=m, x="Timestep", y=metric, label=n, alpha=0.7)
    legend = axis.legend(loc='lower right')
    for line in legend.get_lines():
        line.set_linewidth(2.0)


def importance_plot(t, data, ref_data, ref_norm, axis):
    """
    Plots feature importance of the provided model at the provided timestep, with a shadow of an example droplet behind

    :return:
    """
    plt.cla()
    t -= 1  # to account for alternative numbering of animations
    cols = len(data.columns)
    axis.set_ylim([0.0, 0.5])
    axis.set_xlim([0, len(importance.columns) - 1])

    ref_time = _time_steps()[t]
    ref_row = ref_data.iloc[ref_time, :]
    ref_row = ref_row.div(ref_norm)
    ref_row = ref_row.to_frame().transpose()

    sns.lineplot(x=list(range(0, cols)), y=data.iloc[t, :], color='r')
    lp = sns.lineplot(x=list(range(0, cols)), y=ref_row.iloc[0, :], color='black', alpha=0.2)
    lp.set(ylabel='Coefficient at {T}'.format(T=ref_time), xlabel='Feature Index')
    plt.fill_between(list(range(0, cols)), data.iloc[t, :], color='r', alpha=0.25)
    plt.fill_between(list(range(0, cols)), data.iloc[t, :], ref_row.iloc[0, :], color='black', alpha=0.1)


def covar_heatmap(t, mat, mod, r):
    """
    Produces a heatmap of calculated feature covariances at a provided timestep

    :param t: indexed timestep to observe covariance at
    :param mat: iterable of covariance matrices
    :param mod: model name (for use in title)
    :param r: bounds of pixel values in the heatmap
    :return:
    """
    assert len(r) == 2
    plt.clf()
    t -= 1
    cm = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True, reverse=True)
    sns.heatmap(mat[t], xticklabels=[], yticklabels=[], vmin=r[0], vmax=r[1], cmap=cm)
    plt.title("{m} Cov at {T}".format(m=mod, T=str(_time_steps()[t])))


def write_to_anim(figure, a, f, t, out):
    """
    Defines writer and exports sequence of figures to a gif in the provided directory

    :return:
    """
    # Export importance trends to a gif
    writer = animation.writers['ffmpeg']
    writer = writer(fps=f)
    times = list(range(1, t+1))
    anim = animation.FuncAnimation(figure, a, frames=times, interval=int(1000 / f))
    anim.save(out)


if __name__ == '__main__':
    # settings used from export to gif tutorial; TODO: finetune appearance
    sns.set_context("paper")
    bg_color = "#f1f1f2"
    grid_color = "#bcbabe"
    text_color = "#338494"
    tick_color = "#1995ad"
    rc = {
        'axes.facecolor': bg_color,
        'figure.facecolor': bg_color,
        "text.color": text_color,
        "xtick.color": tick_color,
        "ytick.color": tick_color,
        "axes.edgecolor": grid_color,
        "axes.labelcolor": text_color,
        "grid.color": grid_color,
    }
    sns.set(rc=rc)

    # Setup import & export paths
    args = define_arguments()
    is_processed = True if args.data == "processed" else False

    in_dir = "../logs/{pref}/".format(pref=args.data)
    folders = os.listdir(in_dir)
    folders = sorted([f for f in folders if "txt" not in f])

    out_dir = "../output/figures/"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir = "{od}{pref}/".format(od=out_dir, pref=args.data)  # ensure subdirectories also exists
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Accuracy figure
    acc_models = []
    for model in folders:   # accuracies
        ex = load_model_expr(in_dir + model + "/")
        if "f1" in args.metric:  # manually calculate f1 from precision & recall
            name = re.split("_", args.metric)[0]
            ex[args.metric] = f1(ex, name+"_precision", name+"_recall")
        acc_models.append(ex)
    accuracy_plot(acc_models, folders, args.metric)
    plt.savefig(f'{out_dir}{args.data}_{args.metric}.png')
    plt.clf()

    # Importance Figure
    for model in folders:
        importance_path = in_dir + model + "/importance/"
        if model in ["dt", "logreg"] and args.importance:
            fig, ax = plt.subplots()
            importance = load_model_expr_importances(importance_path, processed=is_processed)
            shadow = _load_droplet_example_shadow()
            animate = partial(importance_plot, data=importance, ref_data=shadow, ref_norm=max(shadow.iloc[0, :])*3,
                              axis=ax)

            fps, time = 1, len(_time_steps())
            gif_name = f'{out_dir}{model}_{args.data}_fi.gif'
            write_to_anim(fig, animate, fps, time, gif_name)

    # Covariance Figure
    for model in folders:  # feature covariances
        importance_path = in_dir + model + "/importance/"
        if model in ["dt", "logreg"] and args.covariance:
            fig, ax = plt.subplots()
            np_files = _load_and_sort(importance_path, ".npy", sort=True)
            cvs = []
            for numpy_file in np_files:
                cov = np.load(importance_path + numpy_file)
                cvs.append(cov)
            vmin, vmax = min([np.min(c) for c in cvs]), max([np.max(c) for c in cvs])
            animate = partial(covar_heatmap, mat=cvs, mod=model, r=(vmin, vmax))

            fps, time = 1, len(_time_steps())
            gif_name = f'{out_dir}{model}_{args.data}_cov.gif'
            write_to_anim(fig, animate, fps, time, gif_name)
