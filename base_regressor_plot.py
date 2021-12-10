from configurations import args
from sklearn.metrics import mean_squared_error

class BaseRegressorPlot():  
  @staticmethod
  def plot_predicted_vs_actual(ax, predicted, actual):
    if args.log_level == 'verbose':
      print('plot_predicted_vs_actual')
    ax.scatter(predicted, actual)
    ax.set_xlabel("Predicted Sales Price ($)")
    ax.set_ylabel("Sales Price ($)")
    ax.set_xlim([0, ax.get_xlim()[1]])
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.plot([-1000000, 1000000], [-1000000, 1000000])
  
  @staticmethod
  def plot_learning_curves(ax, regressor, data_loader, scaled_data=False):
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
  def plot_history_loss(ax, regressor,  inlcule_val_loss=False):
    ax.plot(regressor.get_history_loss(), label='Train Loss')
    if inlcule_val_loss:
      ax.plot(regressor.get_val_history_loss(), label='Val Loss')
    ax.legend()
  
  @staticmethod
  def plot_rmse(ax, regressor,  inlcule_val_loss=False):
    ax.plot(regressor.get_mean_squared_error(), label='Train Error')
    if inlcule_val_loss:
      ax.plot(regressor.get_val_mean_squared_error(), label='Val Error')
    ax.legend()

  @staticmethod
  def clean_out_plot(ax):
    ax.set_axis_off()
  
  @staticmethod
  def plot_error_hist(ax, predicted, actual):
    error = predicted - actual
    ax.hist(error, bins=50)
    ax.set_xlabel("Sales Price Prediction Error ($)")
     

