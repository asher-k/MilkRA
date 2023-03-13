import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import numpy.random as nprand
import sklearn.metrics as met

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.dictionary_based import TemporalDictionaryEnsemble
from sktime.classification.interval_based import RandomIntervalSpectralEnsemble
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.transformations.panel.compose import ColumnConcatenator
from sktime.classification.hybrid import HIVECOTEV2


class Clustering:
    """
    Clustering methods applied to the Droplet dataset. Follows the same class structure as our classification classes,
    with a notable lack of training & testing classes and limited logging.
    """
    def __init__(self):
        self.m = self.models()
        self.samples = None
        self.labels = None

    def models(self):
        """
        Assigns links between command-line model names and their respective functions.

        :return: Dict of corresponding names and functions
        """
        m = {
            "agglomerative": [self.agglomerative]
        }
        return m

    def data(self, d, l):
        """
        Updates model samples & labels for clustering and plotting functions.

        :param d: Droplet data
        :param l: Droplet labels
        """
        self.samples = d
        self.labels = [lab[:4].strip() for lab in l]  # trim class labels for neatness

    def agglomerative(self, **kwargs):
        """
        Agglomerative Hierarchical Clustering method.

        :return: Trained clustering model
        """
        ahc = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
        ahc.fit(self.samples, self.labels)
        return ahc

    def plot_dendrogram(self, model, out_dir, **kwargs):
        """
        Displays a dendrogram of the provided clustering model. Extended from
        https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html.

        :param model: Trained clustering model
        :param out_dir: Output directory
        """
        # Compute linkage matrix
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
        linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)
        # Extra: colour labels according to class
        lab_to_col = {"DBM": "mediumblue", "GTM": "forestgreen", "LBM": "dodgerblue", "LBP+": "goldenrod"}
        ax = plt.gca()
        xlbls = ax.get_xmajorticklabels()
        for lbl in xlbls:
            lbl.set_color(lab_to_col[self.labels[int(lbl.get_text())]])

        # Display with colour labels
        patches = [mpatches.Patch(color=v, label=k) for k, v in lab_to_col.items()]
        ax.legend(handles=patches)
        plt.savefig(f"{out_dir}figs/Dendrogram.png")


class Baselines:
    """
    Non-time series ML baselines. Tracks sample-by-sample prediction of models over multiple runs.
    """
    def __init__(self):
        self.m = self.models()
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.preddict = {}  # Sample-wise tracking here

    def models(self):
        """
        Assigns links between command-line model names and their respective functions.

        :return: Dict of corresponding names and functions
        """
        m = {
            "logreg": [self.logreg],
            "nbayes": [self.nbayes],
            "dt": [self.dtree],
            "knn": [self.knn],
            "svc": [self.svc],
            "mlp": [self.mlp]
        }
        m = _decorate_and_aggregate_models(m)
        return m

    def data(self, tax, tay, tex, tey):
        """
        Updates the training & validation datasets used by the models.

        :param tax: Iterable Training data
        :param tay: Iterable Training labels
        :param tex: Iterable Validation data
        :param tey: Iterable Validation labels
        """
        self.train_x = tax
        self.train_y = tay
        self.test_x = tex
        self.test_y = tey

    def logreg(self, **kwargs):
        """
        Logistic Regression model.
        """
        lr = LogisticRegression(random_state=kwargs['random_state'])  # no convergence at iters=100
        lr.fit(self.train_x, self.train_y)
        return self.predict_and_results(model=lr, **kwargs)

    def nbayes(self, **kwargs):
        """
        Naive Bayes model.
        """
        nb = GaussianNB()
        nb.fit(self.train_x, self.train_y)
        return self.predict_and_results(model=nb, **kwargs)

    def dtree(self, **kwargs):
        """
        Decision Tree Classifier. Verbosity can be enabled in kwargs, which produces a DT visualization for the model.
        This also requires an 'out_dir' kwarg
        """
        state = kwargs["random_state"]
        dt = DecisionTreeClassifier(max_features=min(4, len(self.train_x.columns)),
                                    random_state=state)  # min_impurity_split=0.1
        dt.fit(self.train_x, self.train_y)
        if kwargs['verbose']:  # visualization of DT
            fig = plt.figure(figsize=(20, 20))
            plot_tree(dt, filled=True)
            out_dir = kwargs['out_dir']
            fig.savefig(f"{out_dir}dt_{state}.png")
        return self.predict_and_results(model=dt, **kwargs)

    def knn(self, **kwargs):
        """
        K-Nearest Neighbor algorithm.
        """
        kn = KNeighborsClassifier(n_neighbors=5, weights='distance')
        kn.fit(self.train_x, self.train_y)
        return self.predict_and_results(model=kn, **kwargs)

    def svc(self, **kwargs):
        """
        Support Vector Classifier.
        """
        vec = SVC(kernel='rbf', probability=True)
        vec.fit(self.train_x, self.train_y)
        return self.predict_and_results(model=vec, **kwargs)

    def mlp(self, **kwargs):
        """
        Multilayer Perceptron.
        """
        per = MLPClassifier(hidden_layer_sizes=(32, 16, 8),
                            learning_rate='adaptive',
                            random_state=kwargs["random_state"],
                            max_iter=1000)
        per.fit(self.train_x, self.train_y)
        return self.predict_and_results(model=per, **kwargs)

    def predict_and_results(self, model, **kwargs):
        """
        Provides general functionalities for the prediction & results of trained models, with keywords enabling specific
        functionalities including importance tracking & DT analysis.

        :param model: Trained ML baseline
        :return: statistics on model performance, confusion matrix, feature coefficients, feature splits (DT only)
        """
        preds = model.predict(self.test_x)
        probs = model.predict_proba(self.test_x)
        res, cm = _results_logging(preds, self.test_y, probs, name=str(type(model)))

        importance, splits = None, None
        if kwargs["importance"]:
            if type(model) is LogisticRegression:
                importance = model.coef_[0]
                importance = [v for i, v in enumerate(importance)]
            if type(model) is DecisionTreeClassifier:
                importance = model.feature_importances_
                importance = [v for i, v in enumerate(importance)]
                splits = model.tree_.feature[0]
                if not kwargs["only_acc"]:
                    splits = kwargs['feature_names'][splits]

        # track raw misclassification rates of each sample
        for i, x in enumerate(self.test_x.index):
            ext = (0., 1) if preds[i] == self.test_y[i] else (1., 1)
            self.preddict[x] = tuple(sum(tup) for tup in zip(ext, self.preddict[x]))

        return res, cm, importance, splits


class TSBaselines:
    """
    Time-series baseline models. Note this requires a different approach to preprocessing than non-time series models,
    see data.py/load for details.
    """
    def __init__(self):
        self.m = self.models()
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

    def models(self):
        """
        Assigns links between command-line model names and their respective functions.

        :return: Dict of corresponding names and functions
        """
        m = {"knn": [self.KNN],
             "forest": [self.Forest],
             "tde": [self.tde],
             "rise": [self.rise],
             "shapelet": [self.shapelet],
             "hive": [self.hivecote],
             }
        m = _decorate_and_aggregate_models(m)
        return m

    def data(self, tax, tay, tex, tey):
        """
        Updates the training & validation datasets used by the models.

        :param tax: Iterable Training data
        :param tay: Iterable Training labels
        :param tex: Iterable Validation data
        :param tey: Iterable Validation labels
        """
        self.train_x = tax
        self.train_y = tay
        self.test_x = tex
        self.test_y = tey

    def KNN(self, **kwargs):
        """
        K-Nearest Neighbors Time-series Classifier.
        """
        knn = KNeighborsTimeSeriesClassifier(n_neighbors=5)
        knn.fit(self.train_x, self.train_y)
        return self.predict_and_results(mod=knn, **kwargs)

    def Forest(self, **kwargs):
        """
        Time-Series Forest Classifier.
        """
        steps = [
            ("concatenate", ColumnConcatenator()),  # TSF requires concatenation of columns
            ("classify", TimeSeriesForestClassifier(n_estimators=100, random_state=kwargs["random_state"], n_jobs=-1)),
        ]
        tsf = Pipeline(steps)

        tsf.fit(self.train_x, self.train_y)
        return self.predict_and_results(mod=tsf, **kwargs)

    def tde(self, **kwargs):
        """
        Temporal Dictionary Ensemble (BOSS) classifier.
        """
        cb = TemporalDictionaryEnsemble(n_parameter_samples=250, max_ensemble_size=50, randomly_selected_params=50,
                                        random_state=kwargs["random_state"], time_limit_in_minutes=1)
        cb.fit(self.train_x, self.train_y)
        return self.predict_and_results(mod=cb, **kwargs)

    def rise(self, **kwargs):
        """
        Random Interval Spectral Ensemble Classifier.
        """
        steps = [
            ("concatenate", ColumnConcatenator()),  # TSF requires concatenation of columns
            ("classify", RandomIntervalSpectralEnsemble(random_state=kwargs["random_state"])),
        ]
        r = Pipeline(steps)
        r.fit(self.train_x, self.train_y)
        return self.predict_and_results(mod=r, **kwargs)

    def shapelet(self, **kwargs):
        """
        Shapelet Transform Classifier.
        """
        stc = ShapeletTransformClassifier(random_state=kwargs["random_state"], time_limit_in_minutes=1)
        stc.fit(self.train_x, self.train_y)
        return self.predict_and_results(mod=stc, **kwargs)

    def hivecote(self, **kwargs):
        """
        HIVE-COTE V2 Classifier. Note that performance is poor owing to the ensemble models and a full K=20 validation
        scheme may not be tenable in the short-term.
        """
        hc2 = HIVECOTEV2(n_jobs=-1, random_state=kwargs["random_state"])
        hc2.fit(self.train_x, self.train_y)
        return self.predict_and_results(mod=hc2, **kwargs)

    def predict_and_results(self, mod, **kwargs):
        """
        Provides general functionalities for the prediction & results of trained models.

        :param mod: Trained ML baseline
        :return: statistics on model performance, confusion matrix of model
        """
        preds = mod.predict(self.test_x)
        probs = mod.predict_proba(self.test_x)
        res, cm = _results_logging(preds, self.test_y, probs, name=str(type(mod)))
        return res, cm


def _results_logging(preds, trues, probs=None, name=None,):
    """
    Logs classifier predictions and provides formatted statistics about performance.

    :param preds: Predicted labels
    :param trues: True labels
    :param probs: Predicted label probabilities
    :param name: String identifier of model; deprecated
    :param verbose: Enables logging & confusion matrix
    :return: Tuple of predictive statistics of the model, confusion matrix
    """
    results = {}
    results.update({"Accuracy": met.accuracy_score(trues, preds)})
    results.update({"ROC AUC": np.NaN if probs is None else met.roc_auc_score(trues, probs, multi_class="ovo")})
    class_stats = met.classification_report(trues, preds, output_dict=True, zero_division=0)

    cm = None
    cm = met.confusion_matrix(trues, preds)
    for var_to_save in ['precision', 'recall']:
        [results.update({"{i}_{var}".format(i=i, var=var_to_save): class_stats[i][var_to_save]}) for i in class_stats if
         type(class_stats[i]) is dict]
    results = {k: v for k, v in results.items() if "macro" not in k}  # remove duplicated results
    return results, cm


def _seed_decorator(func):
    """
    Decorator for setting up seed values outside model functions.

    :param func: Function (model) to wrap seed setup
    :return: results of wrapped model
    """
    def execute(*args, **kwargs):
        execute.__name__ = func.__name__  # Name override for exporting
        state = kwargs["random_state"]
        nprand.seed(state)
        return func(*args, **kwargs)
    return execute


def _decorate_and_aggregate_models(m):
    """
    Aggregates baselines to enable one-line testing of all models and decorates each baseline with the seed decorator.
    This method should be run prior to the execution of any baseline to ensure results are truly random and repeatable.

    :param m: Iterable of model functions
    :return: m Appended with the aggregated model functions
    """
    m.update({"all": [i[0] for i in m.values()]})
    [m.update({k: [_seed_decorator(m) for m in m[k]]}) for k in m.keys()]
    return m
