from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


class DataLoader():
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

    def __init__(self, args, should_remove_outliers=True, post_eda=True) -> None:
        self.args = args
        self.cache = {}
        self.df = pd.read_csv(args.data_path)
        if should_remove_outliers:
            self.remove_outliers()
        if post_eda:
                # for features we do not need Id and we need to remove SalesPrice
                # The rest of columns dropped here are ar the result of EDA analysis
            self.df_X = self.df.drop(
                ['SalePrice', 'Id', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                 'LotFrontage', 'GarageType', 'GarageQual', 'GarageFinish',
                 'GarageYrBlt', 'GarageCond', 'BsmtFinType2', 'BsmtExposure',
                 'BsmtQual', 'BsmtCond', 'BsmtFinType1',
                 'PoolQC'], axis=1)
        else:
            self.df_X = self.df.drop(['SalePrice', 'Id'], axis=1)

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

    def get_top_corr_feature(self, top_corr, mi_scores, X_filter_fs):
        features_corr = top_corr.columns.to_list()
        features_mi = mi_scores.index[mi_scores > 0.1].to_list()
        k = []
        k.extend(features_corr)
        k.extend(features_mi)
        features_selected = list(set(k)-{"SalePrice"})
        return features_selected

    # encode the categorical data and fill NAs
    def data_prep(self, x, fillna_va=True):
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

    def get_clean_encoded_data(self, refresh_cache=False):
        ''' This returns clean encoded data 
        It also cache the data to speed up frequent read when cache is 
        enabled
        To refresh cache set the refresh_cashe to True to reload data in 
        cache
        '''
        data = None
        if self.args.enable_cache:
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
        else:
            data = self.clean_encode_data()
        return data

    def clean_encode_data(self):

        encoded_feaures = self.encoding_features()
        return self.split_data(encoded_feaures)

    def split_data(self, working_set):
        ''' Split data into test and train sets

        working_set is the dataset which only contains the features 
        we extracted
        '''
        X_train, X_val, y_train, y_val = train_test_split(
            working_set, self.df_y, test_size=0.10, random_state=1)
        return X_train.toarray(), X_val.toarray(), y_train, y_val

    # Encoding Categorical Features

    def encoding_features(self):
        self.build_features()
        working_set = self.extract_features_org()
        index_of_encoded_cols = [working_set.columns.get_loc(col) for col in [
            'MSSubClass', 'MSZoning', 'Utilities', 'Neighborhood',  'KitchenQual']]
        ct = ColumnTransformer(transformers=[(
            'encoder', OneHotEncoder(), index_of_encoded_cols)], remainder='passthrough')
        return ct.fit_transform(working_set)

    def extract_features_org(self):
        '''This filters out fetures form the raw data based 
        on analysis we at EDA'''
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

    # Dataframe version

    # this section returns the dataframe version
    def extract_features(self, features_list):
        '''This filters out fetures form the raw data based 
            on analysis we did at EDA phase'''
        return self.df[features_list].copy()

    def encode_onehot(df, column_name):
        '''receives a column name are return a dataframe where 
        the column name has been converted to onehot encoding'''
        categorical_values = df[column_name].unique()
        data_to_encode = df.pop(column_name)

        for cat_value in categorical_values:
            col_name = column_name+str(cat_value)
            df[col_name] = (data_to_encode == cat_value) * 1.0

    def clean_encode_data_df(self):
        ''' This return the clean splited data
        returns
            X_train, X_val, y_train, y_val
        '''
        encoded_feaures = self.encoding_features()
        return self.split_data_df(encoded_feaures)

    @staticmethod
    def split_data_df(working_set, combine_back=False):
        ''' Split data into test and train sets
        parameters:
            combine_back
                when True then put back the SalesPrice in to X so can 
                continue to delete rows as needed post analysis
        working_set is the dataset which only contains the features
        we extracted
        '''
        # Take out Sales Price
        y = working_set.pop("SalePrice")
        X_train, X_test, y_train, y_test = train_test_split(
            working_set, y, test_size=0.20, random_state=1)
        # If being used for analysis then
        # combine back the Sales Price to the feature set
        if combine_back:
            X_train["SalePrice"] = y_train
            X_test["SalePrice"] = y_test
        return X_train, X_test, y_train, y_test

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
        return X_train, X_test, y_train, y_test

    @staticmethod
    def get_normalized(X_train, X_test):
        ''' This normilzed data then returns the result
        Returns
            norm_X_train, norm_X_test
        '''
        nomalizer = DataNormalizer(X_train, X_test)
        return nomalizer.get_normilze_data()

    @staticmethod
    def drop_na_from_df(data):
        before = data.shape[0]
        print(f'Before dropping NA {data.shape}')
        data = data.dropna()
        print(
            f'After dropping NA {data.shape}, dropped {before - data.shape[0]}')
        return data

    @staticmethod
    def encode_onehot(data, column_name):
        ''' This onhot encode the categorical columns and drop the original column

        '''
        categorical_values = data[column_name].unique()
        data_to_encode = data.pop(column_name)

        for cat_value in categorical_values:
            col_name = column_name+str(cat_value)
            data[col_name] = (data_to_encode == cat_value) * 1.0

    @staticmethod
    def encode_cat_features(data, features):
        for f in features:
            DataLoader.encode_onehot(data, f)

    @staticmethod
    def is_unique_value_in_cat_features(data, categorical_features):
        ''' This return '''
        is_any_unique_value = False
        for f in categorical_features:
            if (data[f].value_counts() == 1).any():
                for i, v in data[f].value_counts().items():
                    if v == 1:
                        print('unique value:', i, v)
                is_any_unique_value = True
        return is_any_unique_value

    @staticmethod
    def remove_unique_value_of_cat_features(data, categorical_features):
        before = data.shape[0]
        print(f'Before dropping NA {data.shape}')
        for f in categorical_features:
            if (data[f].value_counts() == 1).any():
                remove_list = []
                for i, v in data[f].value_counts().items():
                    if v == 1:
                        print('removing:', i, v)
                        remove_list.append(i)
                data = data[~data[f].isin(remove_list)]
        print(
            f'Before dropping NA {data.shape}, dropped {before - data.shape[0]}')
        return data

    @staticmethod
    def normalize_data(data):
        stats = data.describe().transpose()
        normalized_data = (data - stats['mean']) / stats['std']

        if normalized_data.X_train_norm.isna().values.any():
            raise Exception(
                'Normalization failed! There are' +
                ' features with a mean and a standard deviation equal to zero ')
        return normalized_data

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
        return features_selected


class DataNormalizer:
    def __init__(self, X_train, X_test) -> None:
        self.X_train = X_train
        self.X_test = X_test

    def get_stats(self):
        stats = self.X_train.describe()
        return stats.transpose()

    def normalize(self, data, stats):
        return (data - stats['mean']) / stats['std']

    def get_normilze_data(self):
        # using the same stats for both train and test
        stats = self.get_stats()
        norm_X_train = self.normalize(self.X_train, stats)
        norm_X_test = self.normalize(self.X_test, stats)
        return norm_X_train, norm_X_test
