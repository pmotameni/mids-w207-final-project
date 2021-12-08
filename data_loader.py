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


    # TODO cache this => PM
    def get_raw_split_fs(self):
       '''Return X_train_fs, X_test_fs, y_train_fs, y_test_fs '''

       return train_test_split(
            self.df_X, self.df_y, test_size=0.10, random_state=1)

    def get_top_corr_feature(self, top_corr, mi_scores, X_filter_fs):
        features_corr = top_corr.columns.to_list()
        features_mi = mi_scores.index[mi_scores > 0.1].to_list()
        k = []
        k.extend(features_corr)
        k.extend(features_mi)
        features_selected = list(set(k)-{"SalePrice"})
        #fs_Corr = X_filter_fs[features_selected].corr(
        #).reset_index().melt(id_vars="index")
        #return fs_Corr[(fs_Corr['value'] > 0.6) & (fs_Corr['value'] < 1)]
        return features_selected

    #encode the categorical data and fill NAs
    def data_prep(self, x):
        """
        Encoding Categorical Features, and fill NA

        """
        X = x.copy(deep=True)
        for colname in X.select_dtypes(["object"]):
            X[colname] = X[colname].fillna("noinfo")
            X[colname], _ = X[colname].factorize()
        for colname in X.select_dtypes(["float"]):
            X[colname] = X[colname].fillna(-999999.0)
        return X

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
                if self.args.log_level == 'verbose':
                    print('Cached clean encoded data.')
            else:
                data = cache['clean_encode_data']
                if self.args.log_level == 'verbose':
                    print('Read clean encoded data from cache.')
        if not enable_cache:
            data = self.clean_encode_data()
        return data
    
    # TODO fix this to return the scaled data
    def get_scaled_clean_encoded_data(self, refresh_cache=False):
        ''' This returns scaled version of clean encoded data 
        It also cache the data to speed up frequent read when cache is 
        enabled
        To refresh cache set the refresh_cashe to True to reload data in 
        cache
        '''
        data = None
        enable_cache = self.args.enable_cache
        if enable_cache:
            cache = self.cache
            if not 'scaled_clean_encode_data' in cache or refresh_cache:
                data = self.clean_encode_data()
                cache['scaled_clean_encode_data'] = data
                if self.args.log_level == 'verbose':
                    print('Cached clean encoded data.')
            else:
                data = cache['scaled_clean_encode_data']
                if self.args.log_level == 'verbose':
                    print('Read clean encoded data from cache.')
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
