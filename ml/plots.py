import torch
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

import models
from data import format_name
from functools import partial
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL.Image import Image, fromarray
from sklearn.metrics import ConfusionMatrixDisplay


def _aggregate_image(X, y, agg_type="mean"):
    """
    Computes a mean image from a set of samples.

    :param X: Droplet data
    :param y: Droplet classes
    :param agg_type: Type of aggregation to perform on the set; "mean" computes the mean image, "var" the pixel-wise
    variance
    """
    classes = {"DBM 1000mA Repeats": [], "GTM 1000mA Repeats": [], "LBM 1000mA Repeats": [], "LBP+ 1000mA Repeats": []}
    avg = {"DBM 1000mA Repeats": None, "GTM 1000mA Repeats": None, "LBM 1000mA Repeats": None, "LBP+ 1000mA Repeats": None}
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


def _cam_ups(final_conv, final_dense, top_class, scale=True, size=None):
    """
    Up samples the CAM to the same dimensionality as the original input image and scales CAM values to the (0,255).

    :param final_conv: Input Image convolved by the final layer of convolutional filters
    :param final_dense: Weights of the final dense layer
    :param top_class: Index of the highest-probability class
    :param scale: Bool used to enable scaling of weights to pixel range
    :param size: Tuple shape to up-sample to; should be same dimensionality as our original image
    """
    if size is None:
        size = [31, 131]
    b, c, h, w = final_conv.shape
    cams = []
    for idx in top_class:  # for loop in case we want a CAM for each class
        imd = final_conv.reshape((c, h*w))
        cam = np.matmul(final_dense[idx], imd)  # Matrix multiply the weights of our top class with each filter
        cam = cam.reshape(h, w)
        if scale:  # scales range to 0, 255
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            cam = np.uint8(255 * cam)
        cam = np.array(fromarray(cam).resize(size))
        cams.append(cam)
    return cams


def plot_attention_by_class(model, data, n_layers, out_dir, fname):
    """
    Displays a line plot of the aggregated attention values assigned to each subdivision over each class.

    :param model: ViT model
    :param data: Dataset of droplet samples
    :param n_layers: Int number of attention blocks in the encoder
    :param out_dir: Output directory
    :param fname: File name appended to output
    """
    attentions_by_class = {0: [], 1: [], 2: [], 3: []}
    for sample in data:
        _res, attn = model(torch.unsqueeze(sample[0], dim=0))
        attentions_by_class[sample[1]].append(attn)

    # aggregate
    for k, v in attentions_by_class.items():
        attentions_by_class[k] = [torch.mean(torch.stack([a[i][0] for a in v]), dim=0) for i in range(0, n_layers)]
    flattened_attn = []
    for attn in attentions_by_class.values():  # flatten attentions into a single iterable
        for block in attn:
            flattened_attn.append(block)
    # logging.info(f"Raw mean attention scores: {flattened_attn}")

    y_labels, x_labels = ["DBM", "GTM", "LBM", "LBP+"], [f"Block {i}" for i in range(0, n_layers)]
    fig = plt.figure(figsize=(6., 6.))
    n_rows, n_cols = len(attentions_by_class.keys()), n_layers
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.2)

    for (i, ax), attn in zip(enumerate(grid), flattened_attn):  # Plot over each class/attention block pair
        cutoff = torch.flatten(attn)[torch.sort(torch.flatten(attn))[1][-10].item()]  # Only plot 10 strongest relations
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False, labelsize=6)
        ax.set_yticks(list(range(0, len(attn))))
        ax.set_yticklabels(["<cls>"] + [f"Subdiv {i+1}" for i in range(len(attn)-1)], minor=False, rotation=10)
        for n in ['top', 'right', 'bottom', 'left']:  # removes the bounding box
            ax.spines[n].set_visible(False)
        if i >= (n_rows-1) * n_cols:  # X labels if we are on the final row
            ax.set_xlabel(x_labels[i - ((n_rows-1) * n_cols)])
        if i % n_cols == 0:                # Y labels if we are on the first column
            ax.set_ylabel(y_labels[i // n_cols])

        for xid, x in enumerate(attn):  # Adds connections between all class pairs
            for yid, y in enumerate(x):
                y = y.item() if y.item() >= cutoff else 0
                ax.plot([0, len(attn)], [xid, yid], alpha=y, color="mediumblue", linewidth=2)
    fig.suptitle("Mean Attention by Class & Block")
    fig.supylabel("Class")
    fig.supxlabel("Attention Block")
    plt.savefig(f"{out_dir}Aggr_Classwise_Attn_{fname}.png")


def plot_samplewise_misclassification_rates(m, n_display, labels, arguments, out_dir, mname):
    """
    Displays a horizontal bar chart displaying misclassification rates and classes of samples.

    :param m: Trained Baselines instance, with some results stored in preddict OR iterable of trained PyTorch models
    :param n_display: Number of samples to display in the plot
    :param labels: Set of data labels
    :param arguments: ArgParser used to build the name of the exported figure
    :param out_dir: Export directory
    :param mname: Model name used in file name
    """
    plt.clf()

    # First apply lite preprocessing to obtain relevant rates & samples
    misc_rates = {}
    if type(m) in [models.Baselines, ]:
        misc_rates = {str(k): round(v[0] / v[1], 3) for k, v in m.preddict.items()}
    elif type(m[0]) == torch.nn.Module:
        misc_rates = {}  # TODO: implement sample-wise misclassification for PyTorch
    misc_rates = pd.DataFrame(misc_rates.items(), columns=['id', 'misc']).sort_values('misc', ascending=False)
    misc_rates = misc_rates.loc[:][:n_display]  # often top ~20 perform poorly
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

    fig_name = f"{out_dir}figs/{mname}_misc_rates.png"
    plt.savefig(fig_name)


def plot_confusion_matrix(cm, labels, args, out_dir, mname):
    """
    Plots a confusion matrix aggregated over all experiments of a single model

    :param cm: Pre-computed 2d confusion matrix
    :param labels: Iterable of string labels of data
    :param args: ArgParser object; used to extract experiment details for file name
    :param out_dir: Export directory for the file
    :param mname: Model name displayed in the figure title
    """
    plt.clf()
    cm = np.sum(cm, axis=0)
    cm = np.round(cm / np.sum(cm, axis=1), 3)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted([l[0:4] for l in set(labels)]))
    cm_display.plot()
    fig_name = f"{out_dir}figs/{mname}_cm.png"
    plt.savefig(fig_name)


def plot_mean_vs_mean(X, y, out_dir, agg_type=None):
    """
    Displays heatmaps of the 'average' sample of a class, and identifies major differences between classes.

    :param X: Droplet data
    :param y: Droplet classes
    :param out_dir: Export directory for the file
    :param agg_type: Type of aggregation to perform on the set; "mean" computes the mean image, "var" the pixel-wise
    variance
    """
    _classes, avg = _aggregate_image(X, y, agg_type)
    finals = []
    for k, v1 in avg.items():
        for _, v2 in avg.items():
            if np.allclose(v1, v2):
                finals.append(v1)
            else:
                finals.append(v1-v2)

    fig = plt.figure(figsize=(8., 8.))
    n_rows, n_cols = len(set(y))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.1)
    for i, ax, im in zip(enumerate(finals), grid, finals):
        ax.set_xlabel(list(avg.keys())[i[0] % n_rows][:n_rows])
        ax.set_ylabel(list(avg.keys())[i[0]//n_cols][:n_cols])
        ims = ax.imshow(im, aspect=.12, vmin=-1 if agg_type == "mean" else -0.1, vmax=1 if agg_type == "mean" else 0.1,
                        cmap='coolwarm')
    fig.suptitle(f"\'Pixel-wise\' divergences between mean class samples: {agg_type}")
    fig.supylabel("Temporal Relationships")
    fig.supxlabel("Positional Relationships")
    axins = inset_axes(
        ax,
        width="10%",  # width: 10% of parent_bbox width
        height=f"{str(105*n_cols)}%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    fig.colorbar(ims, cax=axins, ticks=[-1 if agg_type == "mean" else -0.1, 0, 1 if agg_type == "mean" else 0.1])
    plt.savefig(f"{out_dir}figs/Aggr_Mean_Class_Deviations.png")


def plot_sample_vs_mean(X, y, indices, out_dir, plot_type="one"):
    """
    Displays heatmaps identifying differences between the "mean" image of a class and provided samples. Provides the
    option to compare a single sample between its labelled class and all classes.

    :param X: Droplet data
    :param y: Droplet labels
    :param indices: Iterable sample indices from X to compare against the mean class images
    :param out_dir: Export directory for the file
    :param plot_type: Plot comparison type; "one" visualizes only the relevant class, while "all" compares each
    sample against each class
    """
    samples = X[indices, :]
    samples_y = y[indices]
    X = np.delete(X, indices, 0)
    y = np.delete(y, indices, 0)
    _classes, avg = _aggregate_image(X, y, "mean")
    finals = []

    if plot_type == "one":
        for s, s_y in zip(samples, samples_y):  # compute difference between mean image & sample
            s = np.reshape(s, avg[s_y].shape)
            diff = avg[s_y] - s
            finals.append(np.reshape(avg[s_y], (133, 31)))
            finals.append(np.reshape(s, (133, 31)))
            finals.append(np.reshape(diff, (133, 31)))
        fig = plt.figure(figsize=(8., 8.))
        grid = ImageGrid(fig, 111, nrows_ncols=(len(indices), 3), axes_pad=0.1)
        x_labels = ["Mean", "Sample", "Divergence"]
        for i, ax in zip(enumerate(finals), grid):
            ax.set_xlabel(x_labels[i[0] % 3])
            row_index = int(np.floor(1/3 * i[0]))
            ax.set_ylabel(f"{str(indices[row_index])} ({samples_y[row_index][:4].strip()})")
            ims = ax.imshow(i[1], aspect=.12, vmin=-1, vmax=1, cmap='coolwarm')
    elif plot_type == "all":
        for s, s_y in zip(samples, samples_y):  # compute difference between mean image & sample
            finals.append(np.reshape(s, (133, 31)))
            for k, v in avg.items():
                s = np.reshape(s, avg[k].shape)
                diff = avg[k] - s
                finals.append(np.reshape(diff, (133, 31)))
        fig = plt.figure(figsize=(8., 8.))
        grid = ImageGrid(fig, 111, nrows_ncols=(len(indices), 5), axes_pad=0.1)
        x_labels = ["Sample", "DBM", "GTM", "LBM", "LBP+"]
        for i, ax in zip(enumerate(finals), grid):
            ax.set_xlabel(x_labels[i[0] % 5])
            row_index = int(np.floor(1 / 5 * i[0]))
            ax.set_ylabel(f"{str(indices[row_index])} ({samples_y[row_index][:4].strip()})")
            ims = ax.imshow(i[1], aspect=.12, vmin=-1, vmax=1, cmap='coolwarm')

    # Add labels, colorbar, etc
    fig.suptitle("\'Pixel-wise\' divergences between mean class and given samples")
    fig.supylabel("Sample Index, Class")
    fig.supxlabel("Image")
    axins = inset_axes(
        ax,
        width="10%",  # width: 10% of parent_bbox width
        height=f"{str(105 * len(indices))}%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    fig.colorbar(ims, cax=axins, ticks=[-1, 0, 1])
    plt.savefig(f"{out_dir}figs/Aggr_Mean_vs_Samples_{str(indices)}.png")


def plot_epoch_performance(n_epochs, labels, out_dir, mname, *metrics):
    """
    Displays training & validation performance of a DL model over a number of epochs.

    :param n_epochs: Number of epochs used for training
    :param labels: Labels corresponding to each tracked metric; should always be same size as *metrics
    :param out_dir: Export directory for the file
    :param mname: Model name appended during file export; ideally just the index & seed
    """
    eps = list(range(1, n_epochs+1))
    assert len(labels) == len(metrics)

    plt.style.use("ggplot")
    plt.figure()
    for metric, label in zip(metrics, labels):
        plt.plot(eps, metric,  label=label)
    plt.title("Loss/Accuracy by Epoch")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(f"{out_dir}Epochs_{mname}.png")


def plot_embedding_visualization(X, y, out_dir, known_misclassified=None, method="PCA or UMAP"):
    """
    Displays 2d data embeddings obtained via Dimensionality Reduction. Points are colored according to class and
    are labelled according to index within the dataset.

    :param X: Droplet data
    :param y: Droplet classes
    :param out_dir: Export directory for the file
    :param known_misclassified: Iterable of droplet indices; labels for these droplets are highlighted in red
    :param method: str Dimensionality Reduction Technique
    """
    if type(X) is pd.DataFrame:
        X = X.to_numpy()
    # known_misclassified = [6, 9, 25, 32, 44, 61]  # Example; these indices are indeed outliers!
    lab_to_col = {"DBM": "mediumblue", "GTM": "forestgreen", "LBM": "dodgerblue", "LBP+": "goldenrod"}
    colors = [lab_to_col[y[int(i)][:4].strip()] for i, _ in enumerate(X)]  # samples-to-color mapping

    # Plot points
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    ax = plt.gca()
    ax.set_aspect('equal', 'datalim')
    plt.title(f'{method} Dimensionality Reduction', fontsize=16)

    # Add point annotations
    for x_p, y_p, i in zip(X[:, 0], X[:, 1], range(0, len(y))):
        if known_misclassified is not None and i in known_misclassified:
            ax.annotate(i, (x_p, y_p), color="orangered", weight='bold')
        else:
            ax.annotate(i, (x_p, y_p))
    plt.savefig(f"{out_dir}{method}_embedding.png")


def plot_conv_visualizations(t, convs, epochs, title, verbose=False, fig=None, grid=None, n_rows=None, n_cols=None):
    """
    Displays convolutional filters of the provided convolutional layers at a provided timestep. This function is
    leveraged in animate_convolution_by_epoch and will not natively produce any files.

    :param t: Epoch to produce visualizations at (if final, should be 0)
    :param convs: Iterable of convolutional filters across multiple timesteps
    :param epochs: Iterable of the range of epochs
    :param title: Title of the figure
    :param verbose: Enabling verbosity displays the plot
    :param fig: Figure instance (optional)
    :param grid: iterable instance of axes (optional)
    :param n_rows: Number of rows in the image grid
    :param n_cols: Number of columns in the image grid
    """
    t -= 1
    plt.cla()
    weight, to_append = convs[t], epochs[t]
    weight = np.reshape(weight, (weight.shape[0], weight.shape[2], weight.shape[3], weight.shape[1]))
    assert weight.shape[-1] == 1  # input channels must == 1

    if fig is None or grid is None:
        fig = plt.figure(figsize=(8., 8.))
        logging.info(f"Figure shape {n_rows}r {n_cols}c")
        grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.1)
    for i, ax, im in zip(enumerate(weight), grid, weight):
        ims = ax.imshow(im, aspect=1, vmin=-1, vmax=1)
    fig.suptitle(f"Convolutional Filters, {title}:{str(to_append)}")
    if verbose:
        logging.info(f"Shape of conv: {weight.shape}")
        axins = inset_axes(
            ax,
            width="10%",
            height=f"{str(105*len(n_rows))}%",
            loc="lower left",
            bbox_to_anchor=(1.05, 0., 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        fig.colorbar(ims, cax=axins, ticks=[-1, 0, 1])
        plt.show()


def animate_convolution_by_epoch(data, f, t, out_dir, fname, **kwargs):
    """
    Produces an animation of the evolution of convolutional filters in .gif format. Title for the figure can be supplied
    via title=<str>.

    :param data: Convolutional Filter data
    :param f: FPS of the animated gif
    :param t: Number of timesteps in the gif; should be equivalent to len(data)
    :param out_dir: Export directory
    :param fname: Name for the exported .gif file. Ideally just the seed
    """
    shape = data[0].shape
    # Setup writer
    writer = animation.writers['ffmpeg']
    writer = writer(fps=f)
    times = list(range(1, t+1))
    assert len(times) == len(data)

    # Define constant Figure & axes objects
    fig = plt.figure(figsize=(8., 8.))
    n_rows = int(np.ceil(np.sqrt(shape[0])))
    n_cols = int(np.ceil(shape[0] / n_rows))
    logging.info(f"Figure shape {n_rows}r {n_cols}c")
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.1)

    # Generate animation frames & export
    a = partial(plot_conv_visualizations, convs=data, epochs=times, fig=fig, grid=grid,
                title=kwargs["title"] if kwargs["title"] is not None else "", )
    anim = animation.FuncAnimation(fig, a, frames=times, interval=int(1000 / f))
    anim.save(f"{out_dir}Convs_{fname}.gif")
    logging.info(f"Exported convolutions-by-epoch gif to {out_dir}")


def compute_class_activation_maps(model, img, y, out_dir, fname, display=False):
    """
    Displays and exports class activation maps for the provided image and model.

    :param model: CNN model with a GAP layer to extract information required for CAMs
    :param img: Droplet data to produce a CAM of
    :param y: Class for our img
    :param out_dir: Export directory
    :param fname: Name of our CAM file; should include indication of the sample index & seed (if applicable)
    :param display: Saves the figure to the output directory
    :return: 2d NumPy CAM, scaled from 0-255
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

    # Obtain & visualize CAM(s)
    shape = np.squeeze(img, 0).shape
    cam = _cam_ups(convs, final_layer, [ind[0]], size=np.flip(shape), scale=False)
    if display:
        for m in cam:  # iterate over all maps in case we decide to display for each potential class
            plt.cla()
            fig = plt.figure(figsize=(8., 8.))
            grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.1)
            for ax, im, c in zip(grid, [np.squeeze(img), m], ['coolwarm', 'plasma']):
                ims = ax.imshow(im, aspect=1, cmap=c, vmin=-1 if c == 'coolwarm' else None,
                                vmax=1 if c == 'coolwarm' else None)
            fig.suptitle(f"Class Activation Map: True {y} Predicted {ind[0]}")
            plt.savefig(f"{out_dir}CAM_{fname.zfill(3)}.png")
    return cam


def compute_aggregated_cams(model, data, out_dir, aggr_func, fname, display=False):
    """
    Plots class-wise aggregated Class Activation Maps and saves them to the provided output directory.

    :param model: PyTorch model capable of producing CAMs
    :param data: PyTorch Dataset
    :param out_dir: Output directory
    :param aggr_func: Function to use for aggregation; assumed to be np.mean, but can be any NP function which accepts
    array and axis arguments
    :param fname: String appended to filename. Ideally just the seed/index.
    :param display: Saves the figure to the output directory
    :return: Aggregated class CAMs
    """
    cams = {0: [], 1: [], 2: [], 3: []}
    for i, d in enumerate(data):  # CAMs
        x, y = d[0], d[1]
        cam = compute_class_activation_maps(model, x, y, out_dir, str(i))
        cams[y].append(cam)
    aggr_cam = [np.squeeze(aggr_func(v, axis=0)) for k, v in cams.items()]  # compute the mean CAMs by class

    # Finally, display them in a 1x4 plot
    if display:
        vmin, vmax = np.min(aggr_cam), np.max(aggr_cam)

        fig = plt.figure(figsize=(8., 8.))
        grid = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=0.1)
        for ax, (i, mean) in zip(grid, enumerate(aggr_cam)):
            ims = ax.imshow(mean, aspect=1, cmap='coolwarm', vmin=vmin, vmax=vmax)
            ax.set_title(f"Class: {str(i)}")
        axins = inset_axes(ax, width="10%", height="100%", loc="lower left", bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=ax.transAxes, borderpad=0, )  # Color bar
        fig.colorbar(ims, cax=axins, ticks=[vmin, vmax])
        fig.suptitle(f"Class Activation Maps by class {aggr_func.__name__}")
        out = f"{out_dir}CAM_{fname}_{aggr_func.__name__}.png"
        plt.savefig(out)
        logging.info(f"Exported {aggr_func.__name__} CAM images to {out}")
    return aggr_cam


def plot_seed_aggregated_cams(cams, out_dir, aggr_func, fname):
    """
    Plots the aggregations of an iterable of CAM aggregations and saves to an output directory.

    :param cams: Dict of aggregated CAMs
    :param out_dir: Output directory
    :param aggr_func: Function to use for aggregation; assumed to be np.mean, but can be any NP function which accepts
    array and axis arguments
    :param fname: String appended to filename
    """
    cams = np.squeeze(cams)
    final_cams = [aggr_func(cams[:, i], axis=0) for i in range(0, 4)]
    vmin, vmax = np.min(final_cams), np.max(final_cams)
    perc = 90  # Percentile to clip at; 75th percentile can still produce outliers, suggest 90th
    vmax = vmax if vmax-vmin < 10 else np.percentile(final_cams, perc)  # We clip the max if it is sufficiently large

    fig = plt.figure(figsize=(8., 8.))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=0.25)
    for ax, (i, mean) in zip(grid, enumerate(final_cams)):
        ims = ax.imshow(mean, aspect=1, cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax.set_title(f"Class: {str(i)}")
    axins = inset_axes(ax, width="10%", height="100%", loc="lower left", bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=ax.transAxes, borderpad=0, )  # Color bar
    ticks = [vmin, vmax, 0] if vmin < 0 < vmax else [vmin, vmax]
    fig.colorbar(ims, cax=axins, ticks=ticks)
    fig.suptitle(f"Class Activation Maps by class {aggr_func.__name__}, averaged over {str(len(cams))} seeds")
    out = f"{out_dir}figs/CAM_{fname}_{aggr_func.__name__}.png"
    plt.savefig(out)
    logging.info(f"Exported {aggr_func.__name__} CAM images to {out}")


def plot_training_validation_performance(t, v, out_dir):
    """
    Displays a 2d scatterplot of training & validation performances. Provides picture of performance distribution across
    multiple seeds. Credit to unutbu on SOF for the implementation of window-size-agnostic reference line.

    :param t: Iterable training set performances
    :param v: Iterable validation set performances
    :param out_dir: Output directory
    """
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8., 8.))

    # Plot points & reference line
    plt.scatter(x=t, y=v)
    line = lines.Line2D([0.0, 1.0], [0.0, 1.0], color='red')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)

    # Add titles, limits etc
    plt.title("Training vs Validation Performance Distribution")
    plt.xlim(0.0, 1.0)
    plt.xlabel("Training Accuracy")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Validation Accuracy")
    plt.savefig(f"{out_dir}figs/Aggr_Train_vs_Val_Scatter.png")


def plot_training_validation_heatmap(t, v, out_dir, t_size, v_size):
    """
    Displays a 2d heatmap of training & validation performances. Given the low number of samples, a scatterplot contains
    multiple overlaps within a finite space; the heatmap corrects this pitfall by displaying hotspots in addition to
    locations.

    :param t: Iterable training set performances
    :param v: Iterable validation set performances
    :param out_dir: Output directory
    :param t_size: Length of the training set
    :param v_size: Length of the validation set
    """
    fig, ax = plt.subplots(figsize=(8., 8.), ncols=1)
    t, v = [int(i*t_size) for i in t], [int((1-i)*v_size) for i in v]  # scales %-based accuracy to counts
    t_size, v_size = t_size+1, v_size+1  # To account for 100% and 0% accuracy, we plot with an extra count (0)
    img = np.zeros((v_size, t_size))
    for i_t, i_v in zip(t, v):
        img[i_v][i_t] += 1.0

    # Plot heatmap & colorbar
    p = ax.imshow(img, cmap="inferno")
    ax.figure.colorbar(p, ax=ax, shrink=0.5)

    # Set plot parameters
    ax.set_xticks(np.arange(0, t_size, t_size/10), labels=[f"{n*10}%" for n in np.arange(0, 10, 1)], size=12)
    ax.set_yticks(np.arange(0, v_size, v_size/10), labels=reversed([f"{n*10}%" for n in np.arange(1, 11, 1)]), size=12)
    plt.title("Training vs Validation Performance Distribution")
    plt.xlabel("Training Accuracy")
    plt.ylabel("Validation Accuracy")
    plt.savefig(f"{out_dir}figs/Aggr_Train_vs_Val_HM.png")
