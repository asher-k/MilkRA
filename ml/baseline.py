import logging
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy.random as nprand


def log_results(preds, trues, probs, name="?Model?"):
    """
    Logs classifier predictions and provides formatted statistics about performance

    :param preds: predicted labels
    :param trues: true labels
    :param probs: predicted label probabilities
    :param name: string identifier of model
    """
    logging.info("{name} obtained results".format(name=name))
    results = {}
    results.update({"Accuracy": accuracy_score(trues, preds)})
    results.update({"ROC AUC": roc_auc_score(trues, probs, multi_class="ovo")})
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
    res = log_results(preds, y_labels, probs, name="Logistic Regression")
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
    res = log_results(preds, y_labels, probs, name="Naive Bayes")
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
    res = log_results(preds, y_labels, probs, name="Decision Tree")
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


models = {"logreg": [_seed_decorator(logreg)], "nbayes": [_seed_decorator(nbayes)], "dt": [_seed_decorator(dtree)]}
models.update({"all": [_seed_decorator(logreg), _seed_decorator(nbayes), _seed_decorator(dtree)]})
