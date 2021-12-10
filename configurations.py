# this class contains configurable parameters
class args():
    data_path = 'train.csv'
    enable_cache = True
    log_level = 'verbose'  # options: verbose, info
    model_file = 'keras_model.h5'
    resume_nn = True  # When true then loads the save mode if exist
    # To speed up dev if section is in list_of_sections_to_skip
    # it will be skipped
    # full list ['decisontree', 'elasticnet', 'lasso',
    # 'linear',
    # 'neuralnet',
    # 'randomforset', 'ridge']
    list_of_sections_to_skip = ['decisontree', 'elasticnet', 'lasso',
                                'linear',
                                'neuralnet',
                                'randomforset', 'ridge']
    num_epochs = 1000
