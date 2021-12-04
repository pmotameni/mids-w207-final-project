from sklearn.linear_model import Lasso
from base_regressor import BaseRegressor
from configurations import args


class LassoRegressor(BaseRegressor):
    def __init__(self, alpha):
        # this is a Lasso Regressor model
        self.model = Lasso(alpha)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.model.fit(X, y)

    def predict(self, X_test):
        return self.model.predict(X_test)
