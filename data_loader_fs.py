from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


class DataLoaderFS():
    ''' This load and provide data for other modules
    Main instance variables
    df: the raw data

    df_X : the raw feature set
    df_y : the raw lable

    Split of raw data
    X_train
    X_test 
    y_train
    y_test

    '''
    def __init__(self, args, should_remove_outliers = True) -> None:
        self.args = args
        self.cache = {}
        self.df = pd.read_csv(args.data_path)
        if should_remove_outliers:
            self.remove_outliers()
        # for features we do not need Id and we need to remove SalesPrice
        # self.df_X = self.df.drop(['SalePrice', 'Id'], axis=1)
        self.df_X = self.df.drop(
            ['SalePrice', 'Id', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'LotFrontage', 'GarageType', 'GarageQual', 'GarageFinish',
             'GarageYrBlt', 'GarageCond', 'BsmtFinType2', 'BsmtExposure', 
             'BsmtQual', 'BsmtCond', 'BsmtFinType1',
             'PoolQC'], axis=1)
        self.df_y = self.df[['SalePrice']].copy()
        self.set_raw_split()

    def remove_outliers(self):
        self.df = self.df[self.df['GrLivArea'] < 4000]

    def set_raw_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(
                self.df_X, self.df_y, test_size=0.20, random_state=1)

    def get_raw_split(self):
        '''Return X_train, X_test, y_train, y_test '''
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_raw_split_fs(self):
        '''Return X_train, X_test, y_train, y_test '''
        return self.X_train, self.X_test, self.y_train, self.y_test



    def get_top_feature(self, top_corr, mi_scores, X_filter_fs):
        """[summary]version 525

        Args:
            top_corr ([type]): [description]
            mi_scores ([type]): [description]
            X_filter_fs ([type]): [description]

        Returns:
            [type]: [description]
        """
        features_corr = top_corr.columns.to_list()
        features_mi = mi_scores.index.to_list()
        k = []
        k.extend(features_corr)
        k.extend(features_mi)
        features_selected = list(set(k)-{"SalePrice"})
        # fs_Corr = X_filter_fs[features_selected].corr(
        # ).reset_index().melt(id_vars="index")
        # return fs_Corr[(fs_Corr['value'] > 0.6) & (fs_Corr['value'] < 1)]
        return features_selected

    # encode the categorical data and fill NAs
    def data_prep(self, x,fillna_va=True):
        """
        Encoding Categorical Features, and fill NA

        """
        X = x.copy(deep=True)
        for colname in X.select_dtypes(["object"]):
            X[colname] = X[colname].fillna("noinfo")
            X[colname], _ = X[colname].factorize()
        
        for colname in X.select_dtypes(exclude={"object"}):
            X[colname] = X[colname].fillna(0)
        return X

    # Normolize Data (For NN and other SGD models)
    def prep_verify_data_for_nn(self, cat_features, nominal_features):
        ''' Prepare and verify data to be used by Neural Net Regressor
        Retruns:
            X_train, X_test, y_train, y_test
            X_train and X_test are nomalized
        '''
        full_list = nominal_features + cat_features + ['SalePrice']
        df = self.extract_features(full_list)
        df = DataLoader.drop_na_from_df(df)
        df = DataLoader.remove_unique_value_of_cat_features(df, cat_features)
        DataLoader.encode_cat_features(df, cat_features)
        X_train, X_test, y_train, y_test = DataLoader.split_data_df(
            df, combine_back=True)
        X_train, X_test = DataLoader.get_normalized(X_train, X_test)
        return  X_train, X_test, y_train, y_test
