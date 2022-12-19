import numpy as np
import numpy.random as nprand
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


def _results_logging(preds, trues, probs=None, name=None):
    """
    Logs classifier predictions and provides formatted statistics about performance

    :param preds: predicted labels
    :param trues: true labels
    :param probs: predicted label probabilities
    :param name: string identifier of model
    :return: predictive statistics of the model
    """
    results = {}
    results.update({"Accuracy": accuracy_score(trues, preds)})
    results.update({"ROC AUC": np.NaN if probs is None else roc_auc_score(trues, probs, multi_class="ovo")})
    class_stats = classification_report(trues, preds, output_dict=True, zero_division=0)
    for var_to_save in ['precision', 'recall']:
        [results.update({"{i}_{var}".format(i=i, var=var_to_save): class_stats[i][var_to_save]}) for i in class_stats
         if type(class_stats[i]) is dict]
    results = {k: v for k, v in results.items() if "macro" not in k}  # remove duplicated results
    return results


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


class Baselines:
    def __init__(self):
        self.m = self.models()
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

    def models(self):
        """
        Initializes baseline models for evaluation (see Asher for reasoning).
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

        :return:
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
        dt = DecisionTreeClassifier(max_features=4,
                                    random_state=state)  # min_impurity_split=0.1
        dt.fit(self.train_x, self.train_y)
        if kwargs['verbose']:  # visualization of DT
            fig = plt.figure(figsize=(20, 20))
            plot_tree(dt, filled=True, feature_names=kwargs['feature_names'])
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
        Provides general functionalities for the prediction & results of trained models, with keyword enabling specific
        functionalities including importance tracking & DT analysis.

        :param model: trained ML baseline
        :return: statistics on model performance, feature coefficients, feature splits
        """
        preds = model.predict(self.test_x)
        probs = model.predict_proba(self.test_x)
        res = _results_logging(preds, self.test_y, probs, name=str(type(model)))

        importance = None
        splits = None
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

        return res, importance, splits
