from configurations import args
from sklearn.metrics import mean_squared_error

class BaseRegressorPlot():  

  
  @staticmethod
  def plot_predicted_vs_actual(ax, regressor):
    if args.log_level == 'verbose':
      print('get_predicted_vs_actual')
    predicted, actual = regressor.get_predicted_vs_actual()
    ax.scatter(predicted, actual)
    ax.set_xlabel("Predicted Sales Price ($)")
    ax.set_ylabel("Sales Price ($)")
  
  @staticmethod
  def plot_learning_curves(ax, regressor, data_loader, scaled_data=False):
    if scaled_data:
        X_train, X_val, y_train, y_val = data_loader.get_scaled_clean_encoded_data()
    else:
        X_train, X_val, y_train, y_val = data_loader.get_clean_encoded_data()
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        regressor.fit(X_train[:m], y_train[:m])
        y_train_predict = regressor.predict(X_train[:m])
        y_val_predict = regressor.predict(X_val)
        # calculate RMSE
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict,
                                               squared=False))
        val_errors.append(mean_squared_error(
            y_val, y_val_predict, squared=False))
    # ax.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    # ax.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    ax.plot(train_errors, "r-+", linewidth=2, label="train")
    ax.plot(val_errors, "b-", linewidth=3, label="val")
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("RMSE")

  @staticmethod
  def plot_history_loss(ax, regressor,  scaled_data=True):
    # if scaled_data:
    #     X_train, X_val, y_train, y_val = data_loader.get_scaled_clean_encoded_data()
    # else:
    #     X_train, X_val, y_train, y_val = data_loader.get_clean_encoded_data()
    ax.plot(regressor.get_history_loss())
  
  @staticmethod
  def plot_rmse(ax, regressor,  scaled_data=True):
    # if scaled_data:
    #     X_train, X_val, y_train, y_val = data_loader.get_scaled_clean_encoded_data()
    # else:
    #     X_train, X_val, y_train, y_val = data_loader.get_clean_encoded_data()
    ax.plot(regressor.get_mean_squared_error())

  @staticmethod
  def clean_out_plot(ax):
    ax.set_axis_off()
