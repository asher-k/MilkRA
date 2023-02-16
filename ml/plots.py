import torch
from functools import partial

import logging
from PIL.Image import Image, fromarray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
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


def _aggregatae_mean_image(X, y, agg_type=None):
    """
    Produces a mean image from multiple samples
    """
    classes = {"DBM 1000mA Repeats": [], "GTM 1000mA Repeats": [], "LBM 1000mA Repeats": [], "LBP+ 1000mA Repeats": []}
    avg = {"DBM 1000mA Repeats":None, "GTM 1000mA Repeats":None, "LBM 1000mA Repeats":None, "LBP+ 1000mA Repeats":None}
    X = [np.rot90(x, k=3) for x in X]
    for i, x in enumerate(X):
        classes[y[i]].append(x)
    for k, v in classes.items():
        if agg_type == "mean":  # mean image computation
            avg[k] = sum(v) / len(v)
        else:  # pixel-wise variance
            avg[k] = np.array([[np.var([im[r][0][c] for im in v])
                                for c in range(0, len(v[0][0][0]))] for r in range(0, len(v[0]))])
    return classes, avg


def aggregation_differences(X, y, agg_type=None):
    """
    Produces heatmaps of the 'average' sample of a class, and identifies major differences between classes
    """
    classes, avg = _aggregatae_mean_image(X, y, agg_type)
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
        ims = ax.imshow(im, aspect=.12, vmin=-1 if agg_type == "mean" else -0.1, vmax=1 if agg_type == "mean" else 0.1,
                        cmap='coolwarm')
    fig.suptitle(f"\'Pixel-wise\' divergences between mean class samples: {agg_type}")
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


def aggr_vs_sample_difference(X, y, index, agg_type=None, plot_type="one"):
    """
    Produces heatmaps identifying differences between the "mean" image of a class and provided samples
    :param y: sample labels
    :param X: samples
    :param index: iterable of indices extracted from X and compared against mean class images
    :param agg_type: deprecated in method
    :param plot_type: type of plot comparison; "one" visualizes only the relevant class, while "all" compares each
    sample against each class
    """
    samples = X[index, :]
    samples_y = y[index]
    X = np.delete(X, index, 0)
    y = np.delete(y, index, 0)
    classes, avg = _aggregatae_mean_image(X, y, "mean")
    finals = []
    if plot_type == "one":
        for s, s_y in zip(samples, samples_y):  # compute difference between mean image & sample
            s = np.reshape(s, avg[s_y].shape)
            diff = avg[s_y] - s
            finals.append(np.reshape(avg[s_y], (133, 31)))
            finals.append(np.reshape(s, (133, 31)))
            finals.append(np.reshape(diff, (133, 31)))
        fig = plt.figure(figsize=(8., 8.))
        grid = ImageGrid(fig, 111, nrows_ncols=(len(index), 3), axes_pad=0.1)
        x_labels = ["Mean", "Sample", "Divergence"]
        for i, ax in zip(enumerate(finals), grid):
            ax.set_xlabel(x_labels[i[0] % 3])
            row_index = int(np.floor(1/3 * i[0]))
            ax.set_ylabel(f"{str(index[row_index])} ({samples_y[row_index][:4].strip()})")
            ims = ax.imshow(i[1], aspect=.12, vmin=-1, vmax=1, cmap='coolwarm')
    elif plot_type == "all":
        for s, s_y in zip(samples, samples_y):  # compute difference between mean image & sample
            finals.append(np.reshape(s, (133, 31)))
            for k, v in avg.items():
                s = np.reshape(s, avg[k].shape)
                diff = avg[k] - s
                finals.append(np.reshape(diff, (133, 31)))
        fig = plt.figure(figsize=(8., 8.))
        grid = ImageGrid(fig, 111, nrows_ncols=(len(index), 5), axes_pad=0.1)
        x_labels = ["Sample", "DBM", "GTM", "LBM", "LBP+"]
        for i, ax in zip(enumerate(finals), grid):
            ax.set_xlabel(x_labels[i[0] % 5])
            row_index = int(np.floor(1 / 5 * i[0]))
            ax.set_ylabel(f"{str(index[row_index])} ({samples_y[row_index][:4].strip()})")
            ims = ax.imshow(i[1], aspect=.12, vmin=-1, vmax=1, cmap='coolwarm')

    fig.suptitle("\'Pixel-wise\' divergences between mean class and given samples")
    fig.supylabel("Sample Index, Class")
    fig.supxlabel("Image")
    axins = inset_axes(
        ax,
        width="10%",  # width: 10% of parent_bbox width
        height="640%",  # height: 50%
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
    eps = list(range(1, len(epochs["train_loss"])+1))

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(eps, epochs["train_loss"],  label="train_loss")
    plt.plot(eps, epochs["val_loss"], label="val_loss")
    plt.plot(eps, epochs["train_acc"], label="train_acc")
    plt.plot(eps, epochs["val_acc"], label="val_acc")
    plt.title("Loss/Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()


def embedding_visualization(X, y, track_misclassified=True, method=""):
    """
    Displays 2d data embededdings, coloring according to class and labels according to index
    """
    if type(X) is pd.DataFrame:
        X = X.to_numpy()
    lab_to_col = {"DBM": "mediumblue", "GTM": "forestgreen", "LBM": "dodgerblue", "LBP+": "goldenrod"}
    colors = [lab_to_col[y[int(i)][:4].strip()] for i, _ in enumerate(X)]  # samples-to-color mapping

    plt.scatter(X[:, 0], X[:, 1], c=colors)
    ax = plt.gca()
    ax.set_aspect('equal', 'datalim')
    plt.title(f'{method} Dimensionality Reduction', fontsize=16)

    known_misclassified = [6, 9, 25, 32, 44, 61]

    for x_p, y_p, i in zip(X[:, 0], X[:, 1], range(0, len(y))):
        if track_misclassified and i in known_misclassified:
            ax.annotate(i, (x_p, y_p), color="orangered", weight='bold')
        else:
            ax.annotate(i, (x_p, y_p))
    plt.show()


def conv_visualizations(t, convs, epochs, title, verbose=False, fig=None, grid=None):
    """
    Displays convolutional filters of the provided convolutional layers at a provided timestep

    :param t: Epoch to produce visualizations at (if final, should be 0)
    :param verbose: Enabling verbosity displays the plot
    :param fig: Figure instance (optional)
    :param grid: iterable instance of axes (optional)
    """
    t -= 1
    plt.cla()
    weight, to_append = convs[t], epochs[t]
    weight = np.reshape(weight, (weight.shape[0], weight.shape[2], weight.shape[3], weight.shape[1]))

    if fig is None or grid is None:
        fig = plt.figure(figsize=(8., 8.))
        n_rows = int(np.ceil(np.sqrt(weight.shape[0])))
        n_cols = int(np.ceil(weight.shape[0]/n_rows))
        logging.info(f"Figure shape {n_rows}r {n_cols}c")
        grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.1)
    for i, ax, im in zip(enumerate(weight), grid, weight):
        ims = ax.imshow(im, aspect=1, vmin=-1, vmax=1)
    fig.suptitle(f"Convolutional Filters, {title}:{str(to_append)}")
    if verbose:
        logging.info(f"Shape of conv: {weight.shape}")
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


def convolution_by_epoch(data, f, t, out, name, **kwargs):
    """
    Produces an animation of the evolution of convolutional filters in .gif format

    :return:
    """
    # Setup writer
    writer = animation.writers['ffmpeg']
    writer = writer(fps=f)
    times = list(range(1, t+1))
    assert len(times) == len(data)

    # Define constant Figure & axes objects
    fig = plt.figure(figsize=(8., 8.))
    n_rows = int(np.ceil(np.sqrt(data[0].shape[0])))
    n_cols = int(np.ceil(data[0].shape[0] / n_rows))
    logging.info(f"Figure shape {n_rows}r {n_cols}c")
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.1)

    # Generate animation frames & export
    a = partial(conv_visualizations, convs=data, epochs=times, title=kwargs["title"], fig=fig, grid=grid)
    anim = animation.FuncAnimation(fig, a, frames=times, interval=int(1000 / f))
    anim.save(f"{out}/{name}.gif")
    logging.info(f"Exported convolutions by epoch gif to {out}")


def class_activation_maps(model, img, y, out_dir, fname):
    """
    Produces and exports class activation maps for the provided image

    :param model: CNN model with a GAP layer to extract information required for CAMs
    :param img: image to generate CAM
    :param out_dir: output directory
    :return:
    """
    torch_img = torch.unsqueeze(torch.Tensor(img), 0)  # reshape from numpy to [1, ...] Tensor
    pred = torch.squeeze(torch.exp(model(torch_img)[0]))  # Remove from log space
    pred, ind = pred.sort(0, True)  # sort from high->low & obtain indices
    pred = pred.detach().numpy()
    ind = ind.numpy()

    # Obtain final dense layer parameters
    _param = list(model.parameters())
    final_layer = np.squeeze(_param[-2].data.numpy())  # [4, 64]

    # Obtain final convolutional filters
    convs = model(torch_img, early_stopping=True)[0].cpu().detach().numpy()

    # Obtain & visualize CAM
    cam = _cam_upsample(convs, final_layer, [ind[0]], size=[31, 133])

    for map in cam:
        plt.cla()
        fig = plt.figure(figsize=(8., 8.))
        grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.1)
        for ax, im, c in zip(grid, [np.squeeze(img), map], ['coolwarm', 'plasma']):
            ims = ax.imshow(im, aspect=1, cmap=c,
                            vmin=-1 if c == 'coolwarm' else None, vmax=1 if c == 'coolwarm' else None)
        fig.suptitle(f"Class Activation Map: True {y} Predicted {ind[0]}")
        plt.savefig(f"{out_dir}/CAM_{fname}.png")
        # python main.py --type raw --name DELETEME  --experiment classify:dl --load_ranges 0:200:2 200:400:10 400:900:40 --seed 1


def _cam_upsample(final_conv, final_dense, top_class, size=None):
    """
    Upsamples the CAM produced by the NN to be the same dimensionality as the original input image.
    """
    if size is None:
        size = [31, 131]
    b, c, h, w = final_conv.shape
    cams = []
    for idx in top_class:  # for loop in case we want a CAM for each class
        imd = final_conv.reshape((c, h*w))
        cam = np.matmul(final_dense[idx], imd)  # Matrix multiply the weights of our top class with each filter
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        cam = np.array(fromarray(cam).resize(size))
        cams.append(cam)
    return cams

