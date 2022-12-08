from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


def logreg(x_data, x_labels, y_data, y_labels, **kwargs):
    """
    Logistic Regression model

    :param x_data: droplet evaporation sequence training data
    :param x_labels: training labels
    :param y_data: droplet evaporation sequence testing data
    :param y_labels: testing labels
    """
    lr = LogisticRegression(random_state=444, max_iter=1000)
    lr.fit(x_data, x_labels)

    # preds = lr.predict(y_data)
    acc = lr.score(y_data, y_labels)
    print(acc)
    return None


def nbayes(x_data, x_labels, y_data, y_labels, **kwargs):
    """
    Naive Bayes model

    :param x_data: droplet evaporation sequence training data
    :param x_labels: training labels
    :param y_data: droplet evaporation sequence testing data
    :param y_labels: testing labels
    """
    nb = GaussianNB()
    nb.fit(x_data, x_labels)

    acc = nb.score(y_data, y_labels)
    print(acc)
    return None


models = {"logreg": logreg, "nbayes": nbayes}
