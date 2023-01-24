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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


def _results_logging(preds, trues, probs=None, name=None, verbose=False):
    """
    Logs classifier predictions and provides formatted statistics about performance

    :param preds: predicted labels
    :param trues: true labels
    :param probs: predicted label probabilities
    :param name: string identifier of model
    :param verbose: enables logging & confusion matrix
    :return: [predictive statistics of the model, confusion matrix]
    """
    results = {}
    results.update({"Accuracy": met.accuracy_score(trues, preds)})
    results.update({"ROC AUC": np.NaN if probs is None else met.roc_auc_score(trues, probs, multi_class="ovo")})
    class_stats = met.classification_report(trues, preds, output_dict=True, zero_division=0)

    cm = None
    if verbose:
        cm = met.confusion_matrix(trues, preds)
    for var_to_save in ['precision', 'recall']:
        [results.update({"{i}_{var}".format(i=i, var=var_to_save): class_stats[i][var_to_save]}) for i in class_stats if type(class_stats[i]) is dict]
    results = {k: v for k, v in results.items() if "macro" not in k}  # remove duplicated results
    return results, cm


def _seed_decorator(func):
    """
    Decorator for setting up seed values outside of model functions

    :param func: function (model) to wrap seed setup
    :return: results of wrapped model
    """
    def execute(*args, **kwargs):
        state = kwargs["random_state"]
        nprand.seed(state)
        return func(*args, **kwargs)
    return execute


def _decorate_and_aggregate_models(m):
    """
    Aggregates baselines to enable one-line testing of all and decorates each baseline with the seed decorator. This
    method should be run prior to the execution of any baseline to ensure results are truly random and repeatable.

    :return:
    """
    m.update({"all": [i[0] for i in m.values()]})
    [m.update({k: [_seed_decorator(m) for m in m[k]]}) for k in m.keys()]
    return m


class Clustering:
    def __init__(self):
        self.m = self.models()
        self.samples = None
        self.labels = None

    def models(self):
        """
        Initializes clustering methods for evaluation
        """
        m = {
            "agglomerative": [self.agglomerative]
        }
        return m

    def data(self, d, l):
        """
        Updates samples & labels for clustering and plotting functions
        """
        self.samples = d
        self.labels = [lab[:4].strip() for lab in l]  # trim class labels for neatness

    def dendrogram(self, model, **kwargs):
        """
        Creates a linkage matrix and plots a dendrogram of the provided model.

        Extended from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
        """
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

        # extra: colour labels according to class
        lab_to_col = {"DBM": "mediumblue", "GTM": "forestgreen", "LBM": "dodgerblue", "LBP+": "goldenrod"}
        ax = plt.gca()
        xlbls = ax.get_xmajorticklabels()
        for lbl in xlbls:
            lbl.set_color(lab_to_col[self.labels[int(lbl.get_text())]])

        patches = [mpatches.Patch(color=v, label=k) for k, v in lab_to_col.items()]
        ax.legend(handles=patches)
        plt.show()

    def agglomerative(self, **kwargs):
        """
        Agglomerative Hierarchical Clustering
        """
        ahc = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
        ahc.fit(self.samples, self.labels)
        return ahc


class Baselines:
    def __init__(self):
        self.m = self.models()
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

    def models(self):
        """
        Initializes baseline model functions for evaluation (see Asher for hyperparameter reasoning).
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
        Updates train & test datasets used by the models
        """
        self.train_x = tax
        self.train_y = tay
        self.test_x = tex
        self.test_y = tey

    def logreg(self, **kwargs):
        """
        Logistic Regression model
        """
        lr = LogisticRegression(random_state=kwargs['random_state'])  # no convergence at iters=100
        lr.fit(self.train_x, self.train_y)
        return self.predict_and_results(model=lr, **kwargs)

    def nbayes(self, **kwargs):
        """
        Naive Bayes model
        """
        nb = GaussianNB()
        nb.fit(self.train_x, self.train_y)
        return self.predict_and_results(model=nb, **kwargs)

    def dtree(self, **kwargs):
        """
        Decision Tree Classifier
        """
        state = kwargs["random_state"]
        dt = DecisionTreeClassifier(max_features=min(4, len(self.train_x.columns)),
                                    random_state=state)  # min_impurity_split=0.1
        dt.fit(self.train_x, self.train_y)
        if kwargs['verbose']:  # visualization of DT
            fig = plt.figure(figsize=(20, 20))
            plot_tree(dt, filled=True)
            fig.savefig("../output/figures/dt_{ct}.png".format(ct=state))
        return self.predict_and_results(model=dt, **kwargs)

    def knn(self, **kwargs):
        """
        K-Nearest Neighbor algorithm
        """
        kn = KNeighborsClassifier(n_neighbors=5,
                                  weights='distance')
        kn.fit(self.train_x, self.train_y)
        return self.predict_and_results(model=kn, **kwargs)

    def svc(self, **kwargs):
        """
        Support Vector Classifier
        """
        vec = SVC(kernel='rbf', probability=True)
        vec.fit(self.train_x, self.train_y)
        return self.predict_and_results(model=vec, **kwargs)

    def mlp(self, **kwargs):
        """
        Multilayer Perceptron
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

        :param model: trained ML baseline
        :return: statistics on model performance, confusion matrix, feature coefficients, feature splits (DT only)
        """
        preds = model.predict(self.test_x)
        probs = model.predict_proba(self.test_x)
        res, cm = _results_logging(preds, self.test_y, probs, name=str(type(model)), verbose=kwargs['verbose'])

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

        return res, cm, importance, splits
