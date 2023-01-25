import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
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
