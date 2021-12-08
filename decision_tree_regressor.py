from sklearn.tree import DecisionTreeRegressor as DTR
from base_regressor import BaseRegressor
from configurations import args


class DecisionTreeRegressor(BaseRegressor):
    def __init__(self):
        # this is a DecisionTreeRegressor model
        self.model = DTR(random_state=0)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.model.fit(X, y)

    def predict(self, X_test):
        return self.model.predict(X_test)
