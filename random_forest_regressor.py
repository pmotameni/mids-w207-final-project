from sklearn.ensemble import RandomForestRegressor as RFR
from base_regressor import BaseRegressor
from configurations import args


class RandomForestRegressor(BaseRegressor):
    def __init__(self):
        # this is a RandomForestRegressor model
        self.model = RFR(n_estimators=10, random_state=0)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.model.fit(X, y)

    def predict(self, X_test):
        return self.model.predict(X_test)
