import matplotlib.pyplot as plt


class BaseRegressor():
    def calcualte_rmse(self, trueValue=None, predictedValue=None):
        assert trueValue.shape == predictedValue.shape
        return sqrt(mean_squared_error(true_y_log, predicted_y_log))

    def get_predicted_vs_actual(self):
        predicted = self.model.predict(self.X).reshape(-1)
        return predicted, self.y

    def get_mean_squared_error(self):
        # this the abstract method
        pass
