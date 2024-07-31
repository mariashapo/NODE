import numpy as np
import pandas as pd
import jax.numpy as jnp
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d

# TO DO :

# 1) current implementation only allows to load lags within the time segment
# -> 

# 2) currently the data is loaded twice, this does not make sense
# -> implement a DataPreprocessor class that loads data with the spacing in between data points specified
# ---> should it be spacing or just 2 separate start dates?

class DataPreprocessor:
    def __init__(self, file_path, start_date, number_of_points, tau, m, 
                 feature_encoding, target = 'nd',
                 sigma=1, split=300, num_nodes_mult=1, equally_spaced=False,
                 batch_gap = 7):
        self.file_path = file_path
        
        self.start_date = pd.to_datetime(start_date)
        self.number_of_points = number_of_points
        self.sigma = sigma
        self.split = split
        self.num_nodes_mult = num_nodes_mult
        self.equally_spaced = equally_spaced
        
        # expected feature-columns
        self.feature_encoding = feature_encoding
        self.target = target
        
        # lags
        self.prev_week = True 
        self.prev_year = True 
        # m and tau are used for short term embeddings
        self.tau = tau # number of points behind
        self.m = m
        
        # batch_gap expressed in 
        self.batch_gap = batch_gap

    def generate_chebyshev_nodes(self, n, start, end):
        k = np.arange(n)
        x = np.cos(np.pi * k / (n - 1))
        nodes = 0.5 * (end - start) * x + 0.5 * (start + end)
        return np.sort(nodes)

    def load_data(self, offset_days=0):
        """Load data with an offset to accommodate time lags."""
        
        data = pd.read_csv(self.file_path)
        
        data['settlement_date'] = pd.to_datetime(data['settlement_date'])
        
        # obtain the subsample
        data_subsample = data[data['settlement_date'] >= self.start_date][:self.number_of_points]
        # extract hour as a feature
        data_subsample.loc[:,'hour'] = data_subsample['settlement_date'].dt.hour
        
        # extract the relevant columns
        data_subsample.reset_index(drop=True, inplace=True)
        cols = list(self.feature_encoding.keys())
        data_subsample = data_subsample[cols]
        
        # rename the columns
        data_subsample.rename(columns=self.feature_encoding, inplace=True)
        
        data_subsample['var_weekend'] = (data_subsample['t'].dt.dayofweek >= 5).astype(int)
        
        data_subsample = self.add_time_features(data_subsample)
        
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
        data_subsample['t'] = jnp.linspace(0., 1., len(data_subsample))
        return data_subsample
    
    def smooth_signal(self, data, sigma=1):
        for col in data.columns:
            if 'y' == col or 'y' in col:  # Checks if 'y' is part of the column name
                data[col] = gaussian_filter1d(data[col], sigma=sigma)
        return data
    
    def interpolate_data(self, data, t_train, t_test, cols_to_interpolate):
        interpolated_data_train = {}
        interpolated_data_test = {}
        for col in cols_to_interpolate:
            cs = CubicSpline(data['t'], data[col])
            interpolated_data_train[col] = cs(t_train)
            interpolated_data_test[col] = cs(t_test)
        return interpolated_data_train, interpolated_data_test
    
    def preprocess_data(self, data_subsample):
        d = data_subsample.copy()
        t, y = d['t'], d['y']

        # short term embeddings based on the number of points
        for lag in range(1, self.m + 1): 
            days_offset = self.tau * lag 
            offset_date = self.start_date - pd.DateOffset(days=days_offset)
            embedding = self.load_embeddings(adjusted_start_date=offset_date)
            d[f'y_lag{lag}'] = embedding.values 
        
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
        
        d = self.smooth_signal(d, sigma = self.sigma)
        
        t_train, t_test = t[:self.split], t[self.split:]
        
        num_nodes = len(t_train) * self.num_nodes_mult
        num_nodes_test = len(t_test) * self.num_nodes_mult
        
        if self.equally_spaced:
            t_train = np.linspace(t_train.min(), t_train.max(), num_nodes)
            t_test = np.linspace(t_test.min(), t_test.max(), num_nodes_test)
        else:
            t_train = self.generate_chebyshev_nodes(num_nodes, t_train.min(), t_train.max())
            t_test = self.generate_chebyshev_nodes(num_nodes_test, t_test.min(), t_test.max())
        
        data_train, data_test = self.interpolate_data(d, t_train, t_test, d.columns.difference(['t']))
        
        data_train['t'], data_test['t'] = t_train, t_test
        
        df_train, df_test = pd.DataFrame(data_train), pd.DataFrame(data_test)
        
        scaler = StandardScaler()
        columns_to_scale = df_train.columns.difference(['t'])
        df_train[columns_to_scale] = scaler.fit_transform(df_train[columns_to_scale])
        df_test[columns_to_scale] = scaler.transform(df_test[columns_to_scale])
        
        # rearrange columns
        y_columns = [col for col in df_train.columns if col.startswith('y')]
        other_columns = [col for col in df_train.columns if col not in y_columns]
        new_order = y_columns + other_columns
        df_train = df_train[new_order]
        df_test = df_test[new_order]
        
        return df_train, df_test
