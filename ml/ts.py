from baseline import _decorate_and_aggregate_models


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
        m = {}
        m = _decorate_and_aggregate_models(m)
        return m
