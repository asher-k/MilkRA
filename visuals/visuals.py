import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

reshaped_order = ['edge_4_l', 'edge_3_l', '11l', 'edge_2_l', 'edge_1_l', '10l', '9l', '8l', '7l', '6l', '5l', '4l',
                  '3l', '2l', '1l', 'dl_height_midpoint', '1r', '2r', '3r', '4r', '5r', '6r', '7r', '8r', '9r', '10r',
                  'edge_1_r', 'edge_2_r', '11r', 'edge_3_r', 'edge_4_r']


def load_model_expr(path):
    """
    Loads experiment results from a provided directory

    :param path: path to experiment directory
    :return:
    """
    files = os.listdir(path)
    files = [file for file in files if ".csv" in file]
    files.sort(key=lambda f: int(re.split("[_.]", f)[-2]))
    lines = list(reversed([pd.read_csv(path+file).iloc[:, 1:3] for file in files]))
    lines = pd.concat(lines, ignore_index=True)

    lines["Timestep"] = list(reversed(_time_steps()))  # these are hardcoded; remove ASAP
    lines["Variance"] = lines["Accuracy+-"]
    del lines["Accuracy+-"]
    return lines


def load_model_expr_importances(path, processed=False):
    """
    Loads feature importance results from a provided directory

    :param path: path to experiment directory
    :param processed: when "processed" columns are duplicated to maintain symmetry when visualized
    :return:
    """

    files = os.listdir(path)
    files = [file for file in files if ".csv" in file]
    lines = list(reversed([pd.read_csv(path+file).iloc[:, 1:] for file in files]))
    lines = pd.concat(lines, ignore_index=True)
    if processed:
        lines = pd.concat([lines.iloc[:, :], lines.iloc[:, 14::-1]], axis=1)
        lines.columns = [str(i) for i in range(0, len(lines.columns))]
    return lines


def _time_steps():
    return [0,25,50,75,100,150,200,300,400,500,600,700,800,900]


def _load_droplet_example_shadow():
    load_dir = "../data/processed/DBM 1000mA Repeats/221121AP/221121AP_raw.csv"
    shadow_data = pd.read_csv(load_dir)
    shadow_data = shadow_data.iloc[:, 4:]
    shadow_data = shadow_data[reshaped_order]
    return shadow_data


def accuracy_plot(mod, names):
    """
    Plots the accuracy of the provided model

    :return:
    """
    ax = plt.gca()
    ax.set_xlim([0, 900])
    ax.set_ylim([0.0, 1.0])
    for m, n in zip(mod, names):
        sns.lineplot(data=m, x="Timestep", y="Accuracy", label=n, alpha=0.7)
    legend = ax.legend(loc='lower right')
    for line in legend.get_lines():
        line.set_linewidth(2.0)


def importance_plot(time, data, ref_data, ref_norm, axis):
    """
    Plots feature importance of the provided model at the provided timestep, with a shadow of an example droplet behind

    :return:
    """
    plt.cla()
    time -= 1  # to account for alternative numbering of animations
    cols = len(data.columns)
    axis.set_ylim([0.0, 0.5])
    axis.set_xlim([0, len(importance.columns) - 1])

    ref_time = _time_steps()[time]
    ref_row = ref_data.iloc[ref_time, :]
    ref_row = ref_row.div(ref_norm)
    ref_row = ref_row.to_frame().transpose()

    sns.lineplot(x=list(range(0, cols)), y=data.iloc[time, :], color='r')
    t = sns.lineplot(x=list(range(0, cols)), y=ref_row.iloc[0, :], color='black', alpha=0.2)
    t.set(ylabel='Coefficient at {T}'.format(T=ref_time), xlabel='Feature Index')
    plt.fill_between(list(range(0, cols)), data.iloc[time, :], color='r', alpha=0.25)
    plt.fill_between(list(range(0, cols)), data.iloc[time, :], ref_row.iloc[0, :], color='black', alpha=0.1)


if __name__ == '__main__':
    sns.set_context("paper")  # settings used from export to gif tutorial; TODO: finetune appearance
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

    indir = "../logs/"
    outdir = "../output/figures/"
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    pro = False
    pro_str = "processed" if pro else "raw"
    folders = os.listdir(indir)
    folders = sorted([f for f in folders if "txt" not in f])

    acc_models = []
    for model in folders:   # accuracies
        acc_models.append(load_model_expr(indir + model + "/"))
    accuracy_plot(acc_models, folders)
    plt.savefig(f'{outdir}{pro_str}_acc.png')
    plt.clf()

    for model in folders:  # feature importances
        if model in ["dt", "logreg"]:
            fig, ax = plt.subplots()
            importance = load_model_expr_importances(indir + model + "/importance/", processed=pro)
            shadow = _load_droplet_example_shadow()
            animate = partial(importance_plot, data=importance, ref_data=shadow, ref_norm=max(shadow.iloc[0, :])*3,
                              axis=ax)

            # Export importance trends to a gif
            Writer = animation.writers['ffmpeg']
            fps, total_time = 1, 14
            writer = Writer(fps=fps)
            myAnimation = animation.FuncAnimation(fig, animate, frames=list(range(1, 15)), interval=int(1000/fps))
            myAnimation.save(f'{outdir}{model}_{pro_str}_fi.gif')
