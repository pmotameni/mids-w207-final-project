import matplotlib.pyplot as plt
from base_regressor import BaseRegressor
from keras import backend
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input


class NetworkLayer():
    def __init__(self, units, kernel_initializer=None, activation=None):
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.units = units


class NeuralNetworkRegressor(BaseRegressor):
    def __init__(self, layers, input_dim, metrics, loss='mse', optimizer='adam',
                 epochs=100, batch_size=1, shuffle=False, verbose=0):
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.layers = layers
        self.compile(input_dim, loss, metrics)

    def compile(self, input_dim, loss, metrics):
        model = Sequential()
        model.add(Input(shape=(input_dim,)))
        for i in range(len(self.layers)):
            layer = self.layers[i]
            model.add(Dense(layer.units,
                            kernel_initializer=layer.kernel_initializer,
                            activation=layer.activation))
        model.compile(loss=loss, optimizer=self.optimizer, metrics=metrics)
        self.model = model

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.fitHistory = self.model.fit(X, y,
                                         shuffle=self.shuffle,
                                         batch_size=self.batch_size,
                                         verbose=self.verbose, epochs=self.epochs)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_mean_squared_error(self):
        # plt.plot(self.fitHistory.history['root_mean_squared_error'])
        return self.fitHistory.history['root_mean_squared_error']

    def get_history_loss(self):
        return self.fitHistory.history['loss']


# this is one nn network, create more and test them
def create_nn_regressor(X_train, epochs):
    layers = []
    layers.append(NetworkLayer(30, 'normal', 'relu',))
    layers.append(NetworkLayer(10, 'normal', 'relu',))
    layers.append(NetworkLayer(1, 'normal', 'linear'))
    metrics = ['RootMeanSquaredError']
    return NeuralNetworkRegressor(layers, X_train.shape[1],
                                  metrics, epochs=epochs)
