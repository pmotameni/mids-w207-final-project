from os import XATTR_CREATE
import matplotlib.pyplot as plt
from base_regressor import BaseRegressor
from keras import backend
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Input
from configurations import args
from pathlib import Path


class NetworkLayer():
    def __init__(self, units, kernel_initializer=None, activation=None):
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.units = units


class PrintValTrainRatioCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))


class NeuralNetworkRegressor(BaseRegressor):
    def __init__(self, layers, input_dim,  X_valid, y_valid,
                 metrics, loss='mse', optimizer='adam',
                 epochs=100, batch_size=1, shuffle=False, verbose=0):
        self.optimizer = optimizer
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.layers = layers
        if self.can_resume():
            self.resume()
        else:
            self.compile(input_dim, loss, metrics)

    def can_resume(self):
        ''' only resume if resume_nn is enabled and the save model
            exists'''
        can_resume = False
        if args.resume_nn:
            model_file = Path(args.model_file)
            can_resume = model_file.is_file()
        return can_resume

    def resume(self):
        if args.log_level == 'verbose':
            print('Resuming from saved model')
        self.model = load_model(args.model_file)

    def compile(self, input_dim, loss, metrics):
        if args.log_level == 'verbose':
            print('Complied new model')
        model = Sequential()
        model.add(Input(shape=(input_dim,)))
        for i in range(len(self.layers)):
            layer = self.layers[i]
            model.add(Dense(layer.units,
                            kernel_initializer=layer.kernel_initializer,
                            activation=layer.activation))
        model.compile(loss=loss, optimizer=self.optimizer, metrics=metrics)
        self.model = model

    def get_callbacks(self):
        checkpoint_cb = ModelCheckpoint(args.model_file,
                                        save_best_only=True)
        progress_cb = PrintValTrainRatioCallback()
        early_stopping_cb = EarlyStopping(patience=5,
                                          restore_best_weights=True)
        return [checkpoint_cb, progress_cb, early_stopping_cb]

    def fit(self, X, y):

        self.X = X
        self.y = y
        validation_data = (
            self.X_valid, self.y_valid)
        self.fitHistory = self.model.fit(X, y,
                                         shuffle=self.shuffle,
                                         batch_size=self.batch_size,
                                         validation_data=validation_data,
                                         verbose=self.verbose,
                                         epochs=self.epochs,
                                         callbacks=self.get_callbacks())

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_mean_squared_error(self):
        # plt.plot(self.fitHistory.history['root_mean_squared_error'])
        return self.fitHistory.history['root_mean_squared_error']

    def get_history_loss(self):
        return self.fitHistory.history['loss']


# this is one nn network, create more and test them
def create_nn_regressor(X_train,  X_tst, y_tst,  epochs):
    layers = []
    layers.append(NetworkLayer(30, 'normal', 'relu',))
    layers.append(NetworkLayer(10, 'normal', 'relu',))
    layers.append(NetworkLayer(1, 'normal', 'linear'))
    metrics = ['RootMeanSquaredError']
    return NeuralNetworkRegressor(layers, X_train.shape[1],  X_tst, y_tst,
                                  metrics, epochs=epochs)
