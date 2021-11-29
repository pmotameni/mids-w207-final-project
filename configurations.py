# this class contains configurable parameters
class args():
    data_path = 'train.csv'
    enable_cache = True
    log_level = 'verbose'  # options: verbose, info
    model_file = 'keras_model.h5'
    resume_nn = True # When true then loads the save mode if exist

