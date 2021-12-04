from sklearn.linear_model import ElasticNet
from base_regressor import BaseRegressor
from configurations import args


class ElasticNetRegressor(BaseRegressor):
    def __init__(self, alpha, l1_ratio):
        # this is a Elsric Regressor model
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.model.fit(X, y)

    def predict(self, X_test):
        return self.model.predict(X_test)
