import numpy as np
import pandas as pd
import jax.numpy as jnp
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage import gaussian_filter1d

# TO DO :

# currently the data is loaded twice for ADMM, this does not make sense
# -> implement a DataPreprocessor class that loads data with the spacing in between data points specified
# ---> should it be spacing or just 2 separate start dates?

class DataPreprocessor:
    def __init__(self, file_path, start_date, number_of_points, tau, m, 
                 feature_encoding, target = 'nd',
                 prev_hour = False, prev_week = True, prev_year = True,
                 var_weekend = True,
                 split=300, smooth = False, sigma=1):
        self.file_path = file_path
        
        self.start_date = pd.to_datetime(start_date)
        self.number_of_points = number_of_points
        self.sigma = sigma
        self.split = split
        self.smooth = smooth
        
        # expected feature-columns
        self.feature_encoding = feature_encoding
        self.target = target
        self.var_weekend = var_weekend
        
        # lags
        self.prev_week = prev_week 
        self.prev_year = prev_year 
        self.prev_hour = prev_hour
        # m and tau are used for short term embeddings
        self.tau = tau # number of points behind
        self.m = m
    

    def load_data(self):
        
        data = pd.read_csv(self.file_path)
        data['settlement_date'] = pd.to_datetime(data['settlement_date'])
        
        # obtain the subsample
        data_subsample = data[data['settlement_date'] >= self.start_date][:self.number_of_points]
        # extract hour as a feature
        data_subsample.loc[:,'hour'] = data_subsample['settlement_date'].dt.hour.astype(float)
        
        # extract the relevant columns
        data_subsample.reset_index(drop=True, inplace=True)
        cols = list(self.feature_encoding.keys())
        data_subsample = data_subsample[cols]
        
        # rename the columns
        data_subsample.rename(columns=self.feature_encoding, inplace=True)
        
        if self.var_weekend:
            data_subsample['var_weekend'] = (data_subsample['t'].dt.dayofweek >= 5).astype(float)
            
        data_subsample = self.add_time_features(data_subsample)
        
        # print(data_subsample.dtypes)
        return data_subsample
    
    def load_embeddings(self, adjusted_start_date):
        """Load data with an offset to accommodate time lags."""
        
        data = pd.read_csv(self.file_path)
        data['settlement_date'] = pd.to_datetime(data['settlement_date'])
        
        # obtain the subsample
        data_subsample = data[data['settlement_date'] >= adjusted_start_date][:self.number_of_points]
        
        # extract the target column
        y = data_subsample[self.target]
        
        return y

    def add_time_features(self, data_subsample):
        data_subsample['t'] = jnp.linspace(0., 1., len(data_subsample)).astype(float)
        return data_subsample
    
    def smooth_signal(self, data, sigma=1):
        for col in data.columns:
            # filter on the target column or embeddings
            if 'y' == col or 'y_' in col:  
                data[col] = gaussian_filter1d(data[col], sigma=sigma)
        return data
    
    def preprocess_data(self, data_subsample):
        d = data_subsample.copy()
        # short term embeddings based on the number of points
        for lag in range(1, self.m + 1): 
            days_offset = self.tau * lag 
            print(f"Start Data: {self.start_date}")
            print(f"days_offset: {days_offset}")
            print(f"Offset: {self.start_date - pd.DateOffset(days=days_offset)}")
            
            offset_date = self.start_date - pd.DateOffset(days=days_offset)
            embedding = self.load_embeddings(adjusted_start_date=offset_date)
            d[f'y_lag{lag}'] = embedding.values 
        
        if self.prev_hour:
            # last recording
            last_recording_date = self.start_date - pd.DateOffset(hour=1)
            embedding = self.load_embeddings(adjusted_start_date=last_recording_date)
            d[f'y_lag_hout'] = embedding.values
        
        # week embeddings
        if self.prev_week:
            previous_week_date = self.start_date - pd.DateOffset(days=7)
            embedding = self.load_embeddings(adjusted_start_date=previous_week_date)
            d[f'y_lag_week'] = embedding.values 
            
        # year embeddings
        previous_year_date = self.start_date - pd.DateOffset(years=1)
        if self.prev_year:
            previous_year_date = self.start_date - pd.DateOffset(years=1)
            embedding = self.load_embeddings(adjusted_start_date=previous_year_date)
            d[f'y_lag_year'] = embedding.values 
        
        if self.smooth:
            d = self.smooth_signal(d, sigma = self.sigma)
        
        # split the data
        df_train, df_test = d[:self.split], d[self.split:]
        
        # scaling 
        # [1] scale columns containing auxiliary features (non-target/non-embeddings columns)
        scaler_aux = StandardScaler()
        
        excluded_cols = ['t', 'var_weekend', 'y']
        y_cols = [col for col in df_train.columns if 'y_' in col]
        excluded_cols += y_cols # append the embeddings columns
        columns_aux = df_train.columns.difference(excluded_cols)
        
        df_train.loc[:, columns_aux] = scaler_aux.fit_transform(df_train[columns_aux])
        df_test.loc[:, columns_aux] = scaler_aux.transform(df_test[columns_aux])
        
        # [2] scale target and auxiliary columns
        scaler_y = StandardScaler()
        
        debug = False
        if debug:
            y_ = df_train['y'].values.reshape(-1, 1)
            print(f"df_train['y'].values.reshape(-1, 1): {y_.shape}")
            print(f"y_min: {min(y_)}; y_max: {max(y_)}")
        
        # [2.1] scale the target
        df_train.loc[:,'y'] = scaler_y.fit_transform(df_train['y'].values.reshape(-1, 1))
        df_test.loc[:,'y'] = scaler_y.transform(df_test['y'].values.reshape(-1, 1))
        
        # [2.2] scale the embeddings
        for col in y_cols:
            if col == 'y':
                continue
            df_train.loc[:, col] = scaler_y.transform(df_train[col].values.reshape(-1, 1))
            df_test.loc[:, col] = scaler_y.transform(df_test[col].values.reshape(-1, 1))
        
        # reorder columns
        reorder = True
        if reorder:
            y_columns = [col for col in df_train.columns if col.startswith('y')]
            other_columns = [col for col in df_train.columns if col not in y_columns]
            new_order = y_columns + other_columns
            df_train = df_train[new_order]
            df_test = df_test[new_order]
        
        return df_train, df_test
