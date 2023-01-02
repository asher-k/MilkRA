"""
python ts.py
"""
import os
import pandas as pd
import numpy as np
import numpy.random as nprand
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.interval_based import RandomIntervalSpectralEnsemble
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from baseline import _decorate_and_aggregate_models, _results_logging
from main import _col_order


class TSModels:
    def __init__(self):
        self.m = self.models()
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

    def models(self):
        """
        Initializes dict of time-series models for streamlined testing.
        """
        m = {"knn": [self.KNN],
        }
        m = _decorate_and_aggregate_models(m)
        return m

    def data(self, tax, tay, tex, tey):
        """
        Updates train & test datasets used by the models

        :return:
        """
        self.train_x = tax
        self.train_y = tay
        self.test_x = tex
        self.test_y = tey

    def KNN(self, **kwargs):
        """
        K-Nearest Neighbors Time-series Classifier

        :return:
        """
        knn = KNeighborsTimeSeriesClassifier(n_neighbors=5)
        knn.fit(self.train_x, self.train_y)
        return self.predict_and_results(mod=knn, **kwargs)

    def predict_and_results(self, mod, **kwargs):
        """
        Provides general functionalities for the prediction & results of trained models, with keywords enabling specific
        functionalities including importance tracking & DT analysis.

        :param mod: trained ML baseline
        :return: statistics on model performance, confusion matrix, feature coefficients, feature splits (DT only)
        """
        preds = mod.predict(self.test_x)
        probs = mod.predict_proba(self.test_x)
        res, cm = _results_logging(preds, self.test_y, probs, name=str(type(mod)), verbose=kwargs['verbose'])
        return res, cm


def _loadts(data_dir, data_type, **kwargs):
    """
    Loads time series data, performing only minor preprocessing

    :return:
    """
    x, y, norm_consts = [], [], []
    classes = os.listdir(data_dir)
    for c in classes:
        class_dir = "{d}/{c}".format(d=data_dir, c=c)
        seqs, index_cols = [], []
        for f in os.listdir(class_dir + "/"):  # individually load each .csv file
            file_dir = "{c}/{f}/".format(c=class_dir, f=f)
            files = os.listdir(file_dir)
            file = list(filter(lambda fl: data_type + ".csv" in fl, files))
            file = pd.read_csv(f"{file_dir}{file[0]}")
            index_cols.append(file.columns.get_loc("dl_height_midpoint"))  # track normalization index
            seqs.append(file)

        [norm_consts.append(s.iloc[0, i]) for s, i in zip(seqs, index_cols)]  # update normalization constant
        seqs = [s[_col_order(data_type)] for s in seqs]  # reshape column orders
        seqs = [i.iloc[:900, :] for i in seqs]

        if kwargs['features'] is not None:  # reduce features to indicies
            seqs = [i.iloc[:, kwargs['features']] for i in seqs]
        if kwargs['centre_avg']:  # get mean over centre 3 observations and drop original observations
            to_avg = ["dl_height_midpoint", "2l_to_2r", "1l_to_1r"]
            for i in seqs:
                i["midpoints_mean"] = i[to_avg].mean(axis=1)
            seqs = [i.drop(to_avg, axis=1) for i in seqs]

        x += seqs  # note that for pure TS methods we don't flatten
        y += [c] * len(seqs)

    x = [pd.DataFrame(i) for i in x]
    for e, i in enumerate(x):
        i = i.fillna(0)  # 0-imputation
        if kwargs['normalize']:
            x[e] = i.div(norm_consts[e], axis=0)
    return np.array([i.transpose().to_numpy() for i in x]), np.array(y)


# In-built experiment script
if __name__ == "__main__":
    inp = "../data/processed"
    dtype = "processed"

    models = TSModels()
    X, y = _loadts(inp, dtype, normalize=True, features=None, centre_avg=False)
    nprand.seed(0)

    for model in models.m["all"]:
        r, cm = [], []  # results, confusion matricies, feature importance, decision tree splits
        for s in nprand.randint(0, 99999, size=20):
            X_copy = X.copy()
            train_d, test_d, train_l, test_l = train_test_split(X_copy, y, test_size=0.3, stratify=y)
            models.data(train_d, train_l, test_d, test_l)
            result, c = model(random_state=s, verbose=True)
            r.append(result)
            cm.append(c)

        # aggregate seed results & log
        results = pd.DataFrame(r)
        stddevs = results.std()  # light post-run averaging
        results = results.mean(axis=0).round(decimals=3)
        for col, val in stddevs.items():  # reformatting stddev column names
            results[col + "+-"] = val
        results = results.to_frame().transpose()
        results = results.reindex(sorted(results.columns), axis=1)
        print(results)

        # show confusion matrix
        plt.clf()
        cm = np.sum(cm, axis=0)
        cm = np.round(cm / np.sum(cm, axis=1), 3)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                            display_labels=sorted([l[:4] for l in set(y)]))
        cm_display.plot()
        plt.show()
