from typing_extensions import Self
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


class DataLoader():
    def __init__(self, args) -> None:
        self.args = args
        self.cache = {}
        self.df = pd.read_csv(args.data_path)
        # for features we do not need Id and we need to remove SalesPrice
        self.df_X = self.df.drop(['SalePrice', 'Id'], axis=1)
        self.df_y = self.df[['SalePrice']].copy()

    def get_clean_encoded_data(self, refresh_cache=False):
        ''' This returns clean encoded data 
        It also cache the data to speed up frequent read when cache is 
        enabled
        To refresh cache set the refresh_cashe to True to reload data in 
        cache
        '''
        data = None
        enable_cache = self.args.enable_cache
        if enable_cache:
            cache = self.cache
            if not 'clean_encode_data' in cache or refresh_cache:
                data = self.clean_encode_data()
                cache['clean_encode_data'] = data
            else:
                data = cache['clean_encode_data']
                if self.args.log_level == 'verbose':
                    print('Read clean encoded data from cache')
        if not enable_cache:
            data = self.clean_encode_data()
        return data

    def clean_encode_data(self):

        encoded_feaures = self.encoding_features()
        return self.split_data(encoded_feaures)

    def split_data(self, working_set):
        ''' Split data into test and train sets'''
        X_train, X_val, y_train, y_val = train_test_split(
            working_set, self.df_y, test_size=0.10, random_state=1)
        return X_train.toarray(), X_val.toarray(), y_train, y_val

    # Encoding Categorical Features
    def encoding_features(self):
        self.build_features()
        working_set = self.extract_features()
        index_of_encoded_cols = [working_set.columns.get_loc(col) for col in [
            'MSSubClass', 'MSZoning', 'Utilities', 'Neighborhood',  'KitchenQual']]
        ct = ColumnTransformer(transformers=[(
            'encoder', OneHotEncoder(), index_of_encoded_cols)], remainder='passthrough')
        return ct.fit_transform(working_set)

    def extract_features(self):
        list_of_features = ['MSSubClass', 'MSZoning', 'LotArea', 'Utilities',
                            'Neighborhood', 'OverallQual', 'Age', 'RemodAge',
                            'KitchenQual', 'TotalBsmtSF',
                            'GarageCars', 'MoSold', 'GrLivArea', 'total_bath']
        return self.df_X[list_of_features].copy()

    def build_features(self):
        df_X = self.df_X
        # we showed that not considering half-bath as full has better revealing factor
        f_baths = self.get_bath_features_dataset(
            df_X, consider_half_as_full=False)
        df_X['total_bath'] = f_baths['total_bath']
        # Age of house at sale time, remodlled age at sale time
        df_X['Age'] = df_X['YrSold'] - df_X['YearBuilt']
        df_X['RemodAge'] = df_X['YrSold'] - df_X['YearRemodAdd']

    def get_bath_features_dataset(self, x, consider_half_as_full):
        # Set half-bath to half of its value
        bath_props = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']
        bath_dataset = x[['BsmtFullBath',
                          'BsmtHalfBath', 'FullBath', 'HalfBath']].copy()
        if not consider_half_as_full:
            # Total number of bath = number of full + (number of half/2)
            bath_dataset[['BsmtHalfBath', 'HalfBath']] = bath_dataset[[
                'BsmtHalfBath', 'HalfBath']].apply(lambda x: x/2, axis=1)
        bath_dataset['total_bath'] = bath_dataset[bath_props].apply(
            np.sum, axis=1)
        return bath_dataset
