import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from data import format_name
from sklearn.metrics import ConfusionMatrixDisplay


def samplewise_misclassification_rates(baselines, labels, arguments, save_dir, model_name):
    """
    Constructs a horizontal bar chart displaying misclassification rates and classes of the worst-performing samples.
    """
    plt.clf()

    # First apply lite preprocessing to obtain relevant rates & samples
    misc_rates = {str(k): round(v[0] / v[1], 3) for k, v in baselines.preddict.items()}
    misc_rates = pd.DataFrame(misc_rates.items(), columns=['id', 'misc']).sort_values('misc', ascending=False)
    misc_rates = misc_rates.loc[:][:22]  # often top ~20 perform poorly
    misc_rates.reset_index(drop=True, inplace=True)

    lab_to_col = {"DBM": "mediumblue", "GTM": "forestgreen", "LBM": "dodgerblue", "LBP+": "goldenrod"}
    colors = [lab_to_col[labels[int(i)][:4].strip()] for i in misc_rates['id']]  # samples-to-color mapping

    sns.set_context('paper')
    plt.subplots(figsize=(6, 15))
    plt.xlim(0.0, 1.0)
    sns.set_color_codes('pastel')
    sns.barplot(x='misc', y='id', data=misc_rates, palette=colors, edgecolor='w')
    sns.despine(left=True, bottom=True)
    plt.xlabel("Misclassification Rate")
    plt.ylabel("Index")

    # Legend
    ax = plt.gca()
    for i in ax.containers:
        ax.bar_label(i, )
    patches = [mpatches.Patch(color=v, label=k) for k, v in lab_to_col.items()]
    ax.legend(handles=patches, loc='lower right')

    fig_name = format_name(arguments, save_dir, f"_{model_name}_misc_rates.png")
    plt.savefig(fig_name)


def confusion_matrix(cm, labels, arguments, save_dir, model_name):
    """
    Plots a confusion matrix aggregated over all experiments of a single model
    """
    plt.clf()
    cm = np.sum(cm, axis=0)
    cm = np.round(cm / np.sum(cm, axis=1), 3)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted([l[0:4] for l in set(labels)]))
    cm_display.plot()
    fig_name = format_name(arguments, save_dir, f"_{model_name}_cm.png")
    plt.savefig(fig_name)


def aggregation_differences(X, y):
    """
    Produces heatmaps of the 'average' sample of a class, and identifies major differences between classes
    """
    X = [np.rot90(x, k=3) for x in X]
    classes = {"DBM 1000mA Repeats": [], "GTM 1000mA Repeats": [], "LBM 1000mA Repeats": [], "LBP+ 1000mA Repeats": []}
    avg = {"DBM 1000mA Repeats":None, "GTM 1000mA Repeats":None, "LBM 1000mA Repeats":None, "LBP+ 1000mA Repeats":None}
    for i, x in enumerate(X):
        classes[y[i]].append(x)
    for k, v in classes.items():
        avg[k] = sum(v)/len(v)
    finals = []
    for k, v1 in avg.items():
        for _, v2 in avg.items():
            if np.allclose(v1, v2):
                finals.append(v1)
            else:
                finals.append(v1-v2)

    fig = plt.figure(figsize=(8., 8.))
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.1)
    for i, ax, im in zip(enumerate(finals), grid, finals):
        ax.set_xlabel(list(avg.keys())[i[0] % 4][:4])
        ax.set_ylabel(list(avg.keys())[i[0]//4][:4])
        ims = ax.imshow(im, aspect=.12, vmin=-1, vmax=1, cmap='coolwarm')

    fig.suptitle("\'Pixel-wise\' divergences between mean class samples")
    fig.supylabel("Temporal Relationships")
    fig.supxlabel("Positional Relationships")

    axins = inset_axes(
        ax,
        width="10%",  # width: 10% of parent_bbox width
        height="420%",  # height: 50%
        loc="lower left",
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    fig.colorbar(ims, cax=axins, ticks=[1, 2, 3])
    plt.show()


def epoch_performance(epochs):
    """
    Plots training & validation performance of a DL model over a number of epochs
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(epochs["train_loss"], label="train_loss")
    plt.plot(epochs["val_loss"], label="val_loss")
    plt.plot(epochs["train_acc"], label="train_acc")
    plt.plot(epochs["val_acc"], label="val_acc")
    plt.title("Loss/Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()


def conv_visualizations(*convs):
    """
    Displays convolutional filters of the provided convolutional layers
    """
    for conv in convs:
        weight = conv.weight.detach().numpy()
        logging.info(f"Shape of {conv}: {weight.shape}")
        weight = np.reshape(weight, (len(weight) * len(weight[1]), len(weight[1][0]), len(weight[1][0][0]), 1))

        fig = plt.figure(figsize=(8., 8.))
        grid = ImageGrid(fig, 111, nrows_ncols=(len(weight)//5, 5), axes_pad=0.1)
        for i, ax, im in zip(enumerate(weight), grid, weight):
            ims = ax.imshow(im, aspect=1, vmin=-1, vmax=1)

        fig.suptitle(f"Convolutional Filters in {conv}")
        axins = inset_axes(
            ax,
            width="10%",  # width: 10% of parent_bbox width
            height="209%",  # height: 50%
            loc="lower left",
            bbox_to_anchor=(1.05, 0., 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        fig.colorbar(ims, cax=axins, ticks=[1, 2, 3])
        plt.show()
