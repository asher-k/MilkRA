"""
python visuals.py --data processed --metric Accuracy --importance --covariance --name
"""
import functools
import os
import re
from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial


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


def define_arguments():
    """
    Establishes default model & training/inference parameters.

    :return: parsed argument values
    """
    a = ArgumentParser()
    a.add_argument(
        '--name', type=str,
        help='Prefix appended to any generated files'
    )
    a.add_argument(
        '--data', default="processed", type=str, choices=["processed", "raw"],
        help='Type of dataset'
    )
    a.add_argument(
        '--metric', default="Accuracy", type=str,
        help='Name of accuracy metric'
    )
    a.add_argument(
        '--metric_mode', default="timeseries", choices=["timeseries", "independent", "grouped"], type=str,
        help='Display mode for metric results; timeseries plots a line plot, independent a scatter'
    )
    a.add_argument(
        '--importance', default=False, action=BooleanOptionalAction,
        help='Export feature import. figures'
    )
    a.add_argument(
        '--covariance', default=False, action=BooleanOptionalAction,
        help='Export feature covar. figures'
    )
    a.add_argument(
        '--tree_splits', default=False, action=BooleanOptionalAction,
        help='Export DT split figure'
    )
    a = a.parse_args()
    return a


def load_model_expr(path, sort=True):
    """
    Loads experiment results from a provided directory.

    :param path: path to experiment directory
    :param sort: sort filenames to ensure correct ordering
    :return:
    """
    files = _load_and_sort(path, ".csv", sort=sort)
    lines = list(reversed([pd.read_csv(path+file).iloc[:, 1:] for file in files]))
    lines = pd.concat(lines, ignore_index=True)

    lines["Timestep"] = list(reversed(timesteps))
    return lines


def load_model_expr_importances(path, processed=False):
    """
    Loads feature importance results from a provided directory.

    :param path: path to experiment directory
    :param processed: if True columns are duplicated to maintain symmetry when visualized
    :return:
    """
    files = _load_and_sort(path, ".csv")
    lines = list(reversed([pd.read_csv(path+file).iloc[:, 1:] for file in files]))
    lines = pd.concat(lines, ignore_index=True)
    if processed:
        lines = pd.concat([lines.iloc[:, :], lines.iloc[:, 14::-1]], axis=1)
        lines.columns = [str(i) for i in range(0, len(lines.columns))]
    return lines


def load_dt_splits(path):
    """
    Loads split information from logged decison tree performances.

    :param path: Path to decision tree performances
    :return:
    """
    files = _load_and_sort(path, ".csv")
    _lines = list(reversed([pd.read_csv(path+file).iloc[:, 1:] for file in files]))
    _lines = pd.concat(_lines, ignore_index=True, axis=1)

    counts = pd.DataFrame([[len([i for i in _lines[col]if i == feat]) for feat in _col_order("processed")]
                           for col in _lines.columns])
    counts.columns = _col_order("processed")
    return counts


def _load_droplet_example_shadow():
    """
    Loads an example droplet evaporation sequence as a shadow for feature importance analysis

    :return: Iterable representing the shadow of the droplet.
    """
    load_dir = "../data/processed/DBM 1000mA Repeats/221121AP/221121AP_raw.csv"  # Hard-coded example; needs to be tuned
    shadow_data = pd.read_csv(load_dir)
    shadow_data = shadow_data.iloc[:, 4:]
    shadow_data = shadow_data[_col_order("raw")]
    return shadow_data


def _time_steps(path, timeseries=True):
    """
    Infers observed timesteps (and experiments) from observed files.

    :param path: Path to observed timesteps.
    :param timeseries: Determines load order.
    :return: Iterable of observed timesteps
    """
    files = os.listdir(path)
    files = [f for f in files if ".txt" in f]
    if timeseries:
        return [int(re.split("[.]", f)[-2]) for f in files]
    return [re.split("[.]", f)[0] for f in files]


def f1(data, pre, rec):
    """
    F1 score from precision & recall metrics.

    :return: F1 score.
    """
    return 2 * ((data[pre]*data[rec])/(data[pre]+data[rec]))


def metric_timeplot(mod, names, metric):
    """
    Plots the metric of the provided model in a line plot

    :param mod: Model names
    :param names: Name representing information about the model.
    :param metric: Metric to plot.
    """
    axis = plt.gca()
    axis.set_xlim([0, 900])
    axis.set_ylim([0.0, 1.0])
    for m, n in zip(mod, names):
        sns.lineplot(data=m, x="Timestep", y=metric, label=n, alpha=0.7)
    legend = axis.legend(loc='lower right')
    for line in legend.get_lines():
        line.set_linewidth(2.0)


def metric_lineplot(mod, names, metric):
    """
    Plots the metric of the provided model in a non-timeseries lineplot, where each column corresponds to an experiment

    :param mod: Model names
    :param names: Name representing information about the model.
    :param metric: Metric to plot.
    """
    axis = plt.gca()
    axis.set_xlim([0, len(timesteps)-1])
    axis.set_ylim([0.0, 1.0])
    plt.gcf().set_size_inches(1.4*len(timesteps), 4.8)
    for m, n in zip(mod, names):
        s = sns.lineplot(data=m, x="Timestep", y=metric, label=n, alpha=0.7)
    legend = axis.legend(loc='lower right')
    plt.xticks(rotation=20)
    plt.gcf().tight_layout()
    for line in legend.get_lines():
        line.set_linewidth(2.0)


def metric_grouped_pointplot(mod, names, metric):
    """
    Produces several figures of accuracies & associated errors, grouped by model rather than experiment.
    """
    axis = plt.gca()
    sns.set(font_scale=1.2)
    cols = mod[0].columns.to_numpy()
    cols = np.append(cols, "class")
    group_names = ["dt", "knn", "logreg", "mlp", "nbayes", "svc"]
    for m, n in zip(mod, group_names):
        m['class'] = n
    mod = np.array(mod).reshape((24, 26))  # unfortunate hard-coded first argument
    mod = pd.DataFrame(mod, columns=cols)
    print(mod.to_markdown())

    s = sns.catplot(data=mod, kind="bar", x="class", hue="Timestep", y=metric, alpha=0.7)
    s.legend.remove()
    s.set(xlabel=None)
    s.set(ylabel=None)
    plt.ylim(0.0, 1.0)

    plt.gcf().tight_layout()


def importance_plot(t, data, ref_data, ref_norm, axis):
    """
    Plots feature importance of the provided model at the provided timestep, with a shadow of an example droplet behind.

    :return:
    """
    plt.cla()
    t -= 1  # to account for alternative numbering of animations
    cols = len(data.columns)
    axis.set_ylim([0.0, 0.5])
    axis.set_xlim([0, len(importance.columns) - 1])

    ref_time = timesteps[t]
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
    Produces a heatmap of calculated feature covariances at a provided timestep.

    :param t: indexed timestep to observe covariance at
    :param mat: iterable of covariance matrices
    :param mod: model name (for use in title)
    :param r: bounds of pixel values in the heatmap
    """
    assert len(r) == 2
    plt.clf()
    t -= 1
    cm = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True, reverse=True)
    sns.heatmap(mat[t], xticklabels=[], yticklabels=[], vmin=r[0], vmax=r[1], cmap=cm)
    plt.title("{m} Cov at {T}".format(m=mod, T=str(timesteps[t])))


def dt_split_bar(t, d):
    """
    Produces a bar chart of counts of features chosen as the first split point of a Decision Tree.

    :param t: indexed timestep to observe bins at
    :param d: bin count data
    """
    plt.clf()
    plt.ylim((0, 150))
    t -= 1
    cp = sns.barplot(x=_col_order("processed"), y=d.iloc[t, :], order=_col_order("processed"))
    cp.set_xticklabels(cp.get_xticklabels(), rotation=90)
    cp.set(ylabel=None)
    plt.title("First split at {T}".format(T=str(timesteps[t])))
    plt.gcf().tight_layout()


def write_to_anim(figure, a, f, t, out):
    """
    Defines writer and exports sequence of figures to a gif in the provided directory.
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

    # import & export paths
    args = define_arguments()
    is_processed = True if args.data == "processed" else False
    in_dir = "../logs/{pref}/".format(pref=args.data, n=args.name)
    folders = os.listdir(in_dir)
    folders = sorted([f for f in folders if "txt" not in f])
    out_dir = f"../output/figures/{args.data}/{args.name}/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    timesteps = _time_steps(in_dir, args.metric_mode == "timeseries")

    # accuracy figures
    acc_models = []
    for model in folders:
        ex = load_model_expr("{id}{m}/".format(id=in_dir, m=model), args.metric_mode == "timeseries")
        if "f1" in args.metric:  # manually calculate f1 from precision & recall
            name = re.split("_", args.metric)[0]
            ex[args.metric] = f1(ex, f"{name}_precision", f"{name}_recall")
        acc_models.append(ex)
    if args.metric_mode == "timeseries":
        metric_timeplot(acc_models, folders, args.metric)
    elif args.metric_mode == "independent":
        metric_lineplot(acc_models, folders, args.metric)
    else:
        metric_grouped_pointplot(acc_models, folders, args.metric)
    fname = f'{out_dir}{args.name}{args.data}_{args.metric}.png'
    plt.savefig(fname)
    plt.clf()

    # feature importance figures
    for model in folders:
        importance_path = "{id}{m}/importance/".format(id=in_dir, m=model)
        if model in ["dt", "logreg"] and args.importance:
            fig, ax = plt.subplots()
            importance = load_model_expr_importances(importance_path, processed=is_processed)
            shadow = _load_droplet_example_shadow()
            animate = partial(importance_plot, data=importance, ref_data=shadow, ref_norm=max(shadow.iloc[0, :])*3,
                              axis=ax)

            fps, time = 1, len(timesteps)
            gif_name = f'{out_dir}{model}_{args.data}_fi.gif'
            write_to_anim(fig, animate, fps, time, gif_name)

    # Covariance figures
    for model in folders:  # feature covariances
        importance_path = "{id}{m}/importance/".format(id=in_dir, m=model)
        if model in ["dt", "logreg"] and args.covariance:
            fig, ax = plt.subplots()
            np_files = _load_and_sort(importance_path, ".npy", sort=True)
            cvs = []
            for numpy_file in np_files:
                cov = np.load(importance_path+numpy_file)
                cvs.append(cov)
            vmin, vmax = min([np.min(c) for c in cvs]), max([np.max(c) for c in cvs])
            animate = partial(covar_heatmap, mat=cvs, mod=model, r=(vmin, vmax))

            fps, time = 1, len(timesteps)
            gif_name = f'{out_dir}{model}_{args.data}_cov.gif'
            write_to_anim(fig, animate, fps, time, gif_name)

    # DT split figure
    if args.tree_splits:
        split_dir = "{id}dt/importance/splits/".format(id=in_dir)
        fig, ax = plt.subplots(figsize=(6.4, 7.2))
        splits = load_dt_splits(split_dir)

        animate = partial(dt_split_bar, d=splits, f=fig)
        fps, time = 1, len(timesteps)
        gif_name = f'{out_dir}dt_{args.data}_splits.gif'
        write_to_anim(fig, animate, fps, time, gif_name)
