import numpy as np
import pandas as pd
import jax.numpy as jnp
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d

class DataPreprocessor:
    def __init__(self, file_path, start_date, number_of_points, tau, m, sigma=1, split=300, num_nodes_mult=1, equally_spaced=False):
        self.file_path = file_path
        self.start_date = start_date
        self.number_of_points = number_of_points
        self.tau = tau
        self.m = m
        self.sigma = sigma
        self.split = split
        self.num_nodes_mult = num_nodes_mult
        self.equally_spaced = equally_spaced

    def generate_chebyshev_nodes(self, n, start, end):
        k = np.arange(n)
        x = np.cos(np.pi * k / (n - 1))
        nodes = 0.5 * (end - start) * x + 0.5 * (start + end)
        return np.sort(nodes)

    def load_data(self):
        data = pd.read_csv(self.file_path)
        data_subsample = data[data.settlement_date >= self.start_date][:self.number_of_points]
        data_subsample.reset_index(drop=True, inplace=True)
        data_subsample['settlement_date'] = pd.to_datetime(data_subsample['settlement_date'])
        data_subsample.loc[:, 'hour'] = data_subsample['settlement_date'].dt.hour
        data_subsample = data_subsample[['settlement_date', 'temperature', 'hour', 'nd']]
        data_subsample.rename(columns={'settlement_date': 'date', 'temperature': 'var1', 'hour': 'var2', 'nd': 'y'}, inplace=True)
        t = jnp.linspace(0., 1., data_subsample.shape[0])
        data_subsample['t'] = t
        return data_subsample

    def preprocess_data(self, data_subsample):
        d = data_subsample.copy()
        t, y = d['t'], d['y']
        y = gaussian_filter1d(y, sigma=self.sigma)
        d['y'] = y
        
        for i in range(1, self.m + 1):
            d[f'y_lag{i}'] = d['y'].shift(self.tau * i)
        
        first_index = d[f'y_lag{self.m}'].index[~d[f'y_lag{self.m}'].isna()][0]
        self.split -= first_index
        d = d.iloc[first_index:]
        t = d['t']

        t_train, t_test = t[:self.split], t[self.split:]
        
        num_nodes = len(t_train) * self.num_nodes_mult
        if self.equally_spaced:
            t_train = np.linspace(t_train.min(), t_train.max(), num_nodes)
        else:
            t_train = self.generate_chebyshev_nodes(num_nodes, t_train.min(), t_train.max())
        
        interpolated_data_train, interpolated_data_test = {}, {}
        var_cols = [col for col in d.columns if 'var' in col]
        
        for var in var_cols:
            cs = CubicSpline(t, d[var])
            interpolated_data_train[var] = cs(t_train)
            interpolated_data_test[var] = cs(t_test)
        
        cs_y = CubicSpline(t, d['y'])
        interpolated_data_train['y'], interpolated_data_test['y'] = cs_y(t_train), cs_y(t_test)
        
        for i in range(1, self.m + 1):
            cs = CubicSpline(t, d[f'y_lag{i}'])
            interpolated_data_train[f'y_lag{i}'] = cs(t_train)
            interpolated_data_test[f'y_lag{i}'] = cs(t_test)
            for p in range(self.tau * i, len(t_test)):
                interpolated_data_test[f'y_lag{i}'][p] = np.nan
        
        interpolated_data_train['t'], interpolated_data_test['t'] = t_train, t_test
        
        df_train, df_test = pd.DataFrame(interpolated_data_train), pd.DataFrame(interpolated_data_test)
        scaler = StandardScaler()
        columns_to_scale = df_train.columns.difference(['t'])
        df_train[columns_to_scale] = scaler.fit_transform(df_train[columns_to_scale])
        df_test[columns_to_scale] = scaler.transform(df_test[columns_to_scale])
        
        return df_train, df_test
