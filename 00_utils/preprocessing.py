import numpy as np
import pandas as pd
import jax.numpy as jnp

from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d

def generate_chebyshev_nodes(n, start, end):
    k = np.arange(n)
    x = np.cos(np.pi * k / (n - 1))
    nodes = 0.5 * (end - start) * x + 0.5 * (start + end)
    return np.sort(nodes)

def load_data(file_path, start_date, number_of_points):

    data = pd.read_csv(file_path)
    data_subsample = data[data.settlement_date >= start_date]
    # data_subsample = data_subsample[data_subsample.settlement_date < '2010-01-10']
    data_subsample = data_subsample[:number_of_points]
    data_subsample.reset_index(drop=True, inplace=True)

    data_subsample['settlement_date'] = pd.to_datetime(data_subsample['settlement_date'])
    data_subsample.loc[:,'hour'] = data_subsample['settlement_date'].dt.hour

    # select the main columns for the intial testing
    data_subsample = data_subsample[['settlement_date', 'temperature', 'hour', 'nd']]

    print(f"The first/ last time points in the subsample are {np.min(data_subsample.settlement_date)}/ {np.max(data_subsample.settlement_date)}")
    print(f"Covering {np.max(data_subsample['settlement_date'].dt.day) - np.min(data_subsample['settlement_date'].dt.day)} days")

    t = jnp.linspace(0., 1., data_subsample.shape[0]) 

    # How many points cover one day?   
    one_day_map = (data_subsample['settlement_date'].dt.day == np.min(data_subsample['settlement_date'].dt.day))
    n_pt_per_day = one_day_map.sum()
    print(f"Number of points per day: {n_pt_per_day}")

    data_subsample.rename(columns={'settlement_date': 'date', 'temperature': 'var1', 'hour':'var2', 'nd':'y'}, inplace=True)
    data_subsample['t'] = t
    
    return data_subsample

def preprocess_data(data_subsample, tau, m, sigma = 1, split = 300, num_nodes_mult = 1, equally_spaced = False):
    """
    Args:
        data_subsample (dataframe): dataframe containing the subsample of the data;
        expected column names are: 'y', 'date', 'var1', 'var2' ect.
        tau (float): number of points per lag
        m (int): number of lags
    """
    d = data_subsample.copy()   
    columns = data_subsample.columns
    if 't' not in columns:
        raise ValueError("The time column is not present in the dataframe")
    if 'y' not in columns:
        raise ValueError("The target column is not present in the dataframe")
    
    t, y = data_subsample['t'], data_subsample['y']
    
    #----------------------------- SMOOTHING -----------------------------#
    y = gaussian_filter1d(y, sigma = sigma)
    d['y'] = y
    
    #--------------------------------- LAGS -----------------------------#
    for i in range(1, m+1):
        d[f'y_lag{i}'] = d['y'].shift(tau*i)
            
    # the first point that has the last lag available
    first_index = d[f'y_lag{i}'].index[~d[f'y_lag{i}'].isna()][0]

    # drop rows where time lags are not available
    # subtract the first index from the split point 
    split -= first_index
    d = d.iloc[first_index:]
    t = d['t'] 
    
    ##### --------------------------------------------------------------------------------------------------TRAIN TEST SPLIT RATIO #####
    t_train, t_test = t[:split], t[split:]

    print(f"Training data: {t_train.shape[0]} timepoints")
    print(f"Training data: {t_test.shape[0]} timepoints")

    ##### ----------------------------------------------------------------------------------------------TRAIN DATA SIZE MULTIPLIES #####
    # -------------------- CHEBYSHEV NODES FOR THE TRAIN DATA --------------------- #
    if equally_spaced:
        num_nodes = len(t_train)*num_nodes_mult
        t_train = np.linspace(t_train.min(), t_train.max(), num_nodes)
    else:
        num_nodes = len(t_train)*num_nodes_mult
        t_train = generate_chebyshev_nodes(num_nodes, t_train.min(), t_train.max())
    
    # ------------------------- INTERPOLATION FUNCTIONS --------------------------- #

    var_cols_map = ['var' in col for col in d.columns]
    var_cols = d.columns[var_cols_map]
    
    # 1. fit interpolation functions 
    # 2. geneate data
    # 3. save to a new dataframe
    
    interpolated_data_train = {}
    interpolated_data_test = {}
    
    # INDEPENDENT VARIABLES
    for var in var_cols:
        cs = CubicSpline(t, d[var])
        interpolated_data_train[var] = cs(t_train)
        interpolated_data_test[var] = cs(t_test)
        
    # DEPENDENT VARIABLES    
    cs_y = CubicSpline(t, d['y'])
    interpolated_data_train['y'] = cs_y(t_train)
    interpolated_data_test['y'] = cs_y(t_test)
    
    # LAGGED DEPENDENT VARIABLES
    for i in range(1, m+1):
        cs = CubicSpline(t, d[f'y_lag{i}'])
        interpolated_data_train[f'y_lag{i}'] = cs(t_train)
        # here we need some logic to reuse the data available from the training set
        interpolated_data_test[f'y_lag{i}'] = cs(t_test)
        # this gives the whole range, but we need to cut it off where the training data ends
        offset = tau*i # points are needed for the lag
        for p in range(offset, len(t_test)):
            interpolated_data_test[f'y_lag{i}'][p] = np.nan
        
    # TIME
    interpolated_data_train['t'] = t_train
    interpolated_data_test['t'] = t_test

    print(f"Training data: {t_train.shape[0]} timepoints after interpolation")
    
    df_train = pd.DataFrame(interpolated_data_train)
    df_test = pd.DataFrame(interpolated_data_test)

    #------------------------- SCALE ---------------------------#
    scaler = StandardScaler()
    
    columns_to_scale = df_train.columns.difference(['t'])
    df_train[columns_to_scale] = scaler.fit_transform(df_train[columns_to_scale])
    df_test[columns_to_scale] = scaler.transform(df_test[columns_to_scale])
    
    return df_train, df_test