import numpy as np
import numpy.random as nprand
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _results_logging(preds, trues, probs=[], name="?Model?"):
    """
    Logs classifier predictions and provides formatted statistics about performance

    :param preds: predicted labels
    :param trues: true labels
    :param probs: predicted label probabilities
    :param name: string identifier of model
    """
    results = {}
    results.update({"Accuracy": accuracy_score(trues, preds)})
    results.update({"ROC AUC": np.NaN if len(probs) == 0 else roc_auc_score(trues, probs, multi_class="ovo")})
    class_stats = classification_report(trues, preds, output_dict=True, zero_division=0)
    for var_to_save in ['precision', 'recall']:
        [results.update({"{i}_{var}".format(i=i, var=var_to_save): class_stats[i][var_to_save]}) for i in class_stats
         if type(class_stats[i]) is dict]
    results = {k: v for k, v in results.items() if "macro" not in k}  # remove duplicated results
    return results


def logreg(x_data, x_labels, y_data, y_labels, **kwargs):
    """
    Logistic Regression model

    :param x_data: droplet evaporation sequence training data
    :param x_labels: training labels
    :param y_data: droplet evaporation sequence testing data
    :param y_labels: testing labels
    :return: statistics on model performance
    """
    lr = LogisticRegression(random_state=kwargs['random_state'], max_iter=100)  # no convergence at iters=100
    lr.fit(x_data, x_labels)

    preds = lr.predict(y_data)
    probs = lr.predict_proba(y_data)
    res = _results_logging(preds, y_labels, probs, name="Logistic Regression")
    return res


def nbayes(x_data, x_labels, y_data, y_labels, **kwargs):
    """
    Naive Bayes model

    :param x_data: droplet evaporation sequence training data
    :param x_labels: training labels
    :param y_data: droplet evaporation sequence testing data
    :param y_labels: testing labels
    :return: statistics on model performance
    """
    nb = GaussianNB()
    nb.fit(x_data, x_labels)

    preds = nb.predict(y_data)
    probs = nb.predict_proba(y_data)
    res = _results_logging(preds, y_labels, probs, name="Naive Bayes")
    return res


def dtree(x_data, x_labels, y_data, y_labels, **kwargs):
    """
    Decision Tree Classifier

    :param x_data: droplet evaporation sequence training data
    :param x_labels: training labels
    :param y_data: droplet evaporation sequence testing data
    :param y_labels: testing labels
    :return: statistics on model performance
    """
    dt = DecisionTreeClassifier(random_state=kwargs["random_state"])
    dt.fit(x_data, x_labels)

    preds = dt.predict(y_data)
    probs = dt.predict_proba(y_data)
    res = _results_logging(preds, y_labels, probs, name="Decision Tree")
    return res


def knn(x_data, x_labels, y_data, y_labels, **kwargs):
    """
    K-Nearest Neighbor algorithm

    :param x_data: droplet evaporation sequence training data
    :param x_labels: training labels
    :param y_data: droplet evaporation sequence testing data
    :param y_labels: testing labels
    :return: statistics on model performance
    """
    kn = KNeighborsClassifier()
    kn.fit(x_data, x_labels)

    preds = kn.predict(y_data)
    probs = kn.predict_proba(y_data)
    res = _results_logging(preds, y_labels, probs, name="KNN")
    return res


def svc(x_data, x_labels, y_data, y_labels, **kwargs):
    """
    Support Vector Classifier

    :param x_data: droplet evaporation sequence training data
    :param x_labels: training labels
    :param y_data: droplet evaporation sequence testing data
    :param y_labels: testing labels
    :return: statistics on model performance
    """
    vec = SVC()
    vec.fit(x_data, x_labels)

    preds = vec.predict(y_data)
    res = _results_logging(preds, y_labels, name="SVC")
    return res


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


models = {"logreg": [logreg], "nbayes": [nbayes], "dt": [dtree], "knn": [knn], "svc": [svc]}
models.update({"all": [i[0] for i in models.values()]})
[models.update({k: [_seed_decorator(m) for m in models[k]]}) for k in models.keys()]