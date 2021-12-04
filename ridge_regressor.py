from sklearn.linear_model import Ridge
from base_regressor import BaseRegressor
from configurations import args


class RidgeRegressor(BaseRegressor):
    def __init__(self):
        # this is a Ridge Regressor model
        self.model = Ridge()

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.model.fit(X, y)

    def predict(self, X_test):
        return self.model.predict(X_test)
