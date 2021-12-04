from sklearn.linear_model import LinearRegression
from base_regressor import BaseRegressor
from configurations import args


class LinearRegressor(BaseRegressor):
    def __init__(self):
        # this is linear Regression model
        self.model = LinearRegression()

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.model.fit(X, y)

    def predict(self, X_test):
        return self.model.predict(X_test)
