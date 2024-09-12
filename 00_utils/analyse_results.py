import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import importlib
import ast
import jax.numpy as jnp
from jaxlib import xla_extension as jax_types

class Graphs:
    @staticmethod
    def plot_boxplots(data1, data2, labels, title, ylabel, colors=('blue', 'green'), color_labels = ['Pyomo', 'Diffrax'],
                    x_label = 'Model Size Configuration', y_log = True):
        n_groups = len(data1)
        positions_1 = [2 * i + 1.2 for i in range(n_groups)]
        positions_2 = [2 * i + 1.8 for i in range(n_groups)]
        
        plt.figure(figsize=(10, 6))
        box1 = plt.boxplot(data1, positions=positions_1, widths=0.5, patch_artist=True, boxprops=dict(facecolor=colors[0]))
        box2 = plt.boxplot(data2, positions=positions_2, widths=0.5, patch_artist=True, boxprops=dict(facecolor=colors[1]))
        
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(ylabel)
        if y_log:
            plt.yscale('log')
        
        xticks = [2 * i + 1.5 for i in range(n_groups)]
        plt.xticks(ticks=xticks, labels=labels)
        
        patch1 = mpatches.Patch(color=colors[0], label=color_labels[0])
        patch2 = mpatches.Patch(color=colors[1], label=color_labels[1])
        
        plt.legend(handles=[patch1, patch2], loc='upper left')
        plt.grid(True)
        plt.show()
        
    @staticmethod
    def plot_single_boxplot(data, labels, title, ylabel, color='blue', label = 'Data Label',
                            x_label = 'Model Size Configuration', y_log = True):
        n_groups = len(data)
        positions = [i + 1 for i in range(n_groups)]
        
        plt.figure(figsize=(10, 6))
        box = plt.boxplot(data, positions=positions, widths=0.6, patch_artist=True, boxprops=dict(facecolor=color))
        
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(ylabel)
        if y_log:
            plt.yscale('log')
        
        plt.xticks(ticks=positions, labels=labels)
        
        patch = mpatches.Patch(color=color, label=label)  
        
        # plt.axhline(y=min_acc, color='r', linestyle='--', label='Minimum MSE')
        plt.legend(handles=[patch], loc='upper left')
        plt.grid(True)
        plt.show()



class Graphs_training:
    def __init__(self):
        self.regular_pre_time = None
        self.pyomo_pre_time = None
        self.regular_pre_time_pt = None
        self.pyomo_pre_time_pt = None
    
    def set_pretraining_time(self, pretraining_time, type):
        
        if type == 'regular':
            self.regular_pre_time = pretraining_time
        elif type == 'pyomo':
            self.pyomo_pre_time = pretraining_time
        elif type == 'pt_regular':
            self.regular_pre_time_pt = pretraining_time
        elif type == 'pt_pyomo':
            self.pyomo_pre_time_pt = pretraining_time
        else:
            raise ValueError(f"Unknown pre-training type '{type}'.")
    
    @staticmethod
    def plot_training_losses(df, title, ylabel, x_label='Epochs', y_log=True):
        plt.figure(figsize=(10, 6))
        for i, row in df.iterrows():
            plt.plot(row['training_loss'], label=row.name)
        
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(ylabel)
        if y_log:
            plt.yscale('log')
        
        plt.legend()
        plt.grid(True)
        plt.show()

        
    @staticmethod
    def extract_training_loss(df, loss_col='training_loss', index_col='pretraining'):
        """
        Ensure 'training_loss' and 'pretraining' columns are present in the DataFrame.
        """
        def extract_training_loss_row(row):
        # check if the 'training_loss' is not empty and has the required index
            if isinstance(row[loss_col], list) and len(row[loss_col]) > row['index']:
                return row[loss_col][row['index']]
                # if no pre-training 0th index is the full training
                # if there is pre-training 0th index is the pre-training and 1st index is the full training
            return None  

        df['index'] = df[index_col].astype(int)  # Convert True/False to 1/0

        df[loss_col] = df.apply(extract_training_loss_row, axis=1)
        # drop temporary index
        df.drop(columns=['index'], inplace=True)
        return df
    
    @staticmethod
    def split_train_test_losses(df):
        """
        Splits the training and testing losses into separate columns.
        """
        # ensure both training and testing losses are extracted from the original data structure
        df['training_loss'], df['testing_loss'] = zip(*df['training_loss'].apply(lambda x: (x[0], x[1]) if len(x) > 1 else (None, None)))

        return df
    
    @staticmethod
    def prepare_timings_general(df):
        """
        Prepares the timings for plotting.
        """
        n_epochs = df['training_loss'].apply(len)
        
        # calculate time per epoch; use element-wise division
        t_per_epoch = df['time_elapsed'] / n_epochs
        
        # calculate times for each row using the previously calculated n_epochs and t_per_epoch
        df['times'] = df.apply(lambda row: np.arange(n_epochs.loc[row.name]) * t_per_epoch.loc[row.name], axis=1)    
        
        return df
    
    def prepare_timings(self, df):
        """
        Prepares the timings for plotting.
        """
        n_epochs = df['training_loss'].apply(len)
        
        # calculate time per epoch; use element-wise division
        t_per_epoch = df['time_elapsed'] / n_epochs
        
        # calculate times for each row using the previously calculated n_epochs and t_per_epoch
        df['times'] = df.apply(lambda row: np.arange(n_epochs.loc[row.name]) * t_per_epoch.loc[row.name], axis=1)    
        
        if self.regular_pre_time:
            map_regular = (df.type == 'jd') & (df.pretraining == True)
            df.loc[map_regular, 'times'] = df.loc[map_regular, 'times'] + self.regular_pre_time
        
        if self.pyomo_pre_time:
            map_pyomo = (df.type == 'jd') & (df.pyomo_pretraining == True)
            df.loc[map_pyomo, 'times'] = df.loc[map_pyomo, 'times'] + self.pyomo_pre_time
        
        if self.regular_pre_time_pt:
            map_regular_pt = (df.type == 'pt') & (df.pretraining == True)
            df.loc[map_regular_pt, 'times'] = df.loc[map_regular_pt, 'times'] + self.regular_pre_time_pt
            
        if self.pyomo_pre_time_pt:
            map_pyomo_pt = (df.type == 'pt') & (df.pyomo_pretraining == True)
            df.loc[map_pyomo_pt, 'times'] = df.loc[map_pyomo_pt, 'times'] + self.pyomo_pre_time_pt
        
        return df

class Results:
    @staticmethod
    def load_results(file_path):
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        return results
    
    @staticmethod
    def key_sample(data, n = 0):
        """
        Returns a sample key from a dictionary.
        """
        return [k for k in data.keys()][n]
    
    @staticmethod
    def parse_results(results, keys_list):
        """
        Parses results into a DataFrame.
        """
        records = []
        for key, values in results.items():
            
            if has_nested_tuple(key):
                unpack_key = True
            else:
                unpack_key = False
                
            if unpack_key:
                (key, date) = key
            # ensure the key is a string
            if isinstance(key, tuple):
                record = dict(zip(keys_list, key))
            else:
                # evaluate the string to a dictionary
                record = ast.literal_eval(key)
            
            if unpack_key:
                record['date'] = date
            # merge key - value pairs
            record.update(values)
            # append the record to the list
            records.append(record)

        df = pd.DataFrame(records)
        return df
    
    @staticmethod
    def to_numpy(array):
        """Converts JAX array to NumPy array if necessary."""
        if isinstance(array, jnp.ndarray):
            return np.array(array)
        return array
    
    @staticmethod
    def group_into_lists(df, group_col, value_col):
        """
        Groups a DataFrame into lists.
        """
        df_copy = df.copy()
        # check if jax types are present in the DataFrame
        df_copy = df_copy.map(Results.to_numpy)
        grouped = df_copy.groupby(group_col).agg(list)[value_col]
        return grouped
    
    @staticmethod
    def collect_data(results, custom_names=None):
        """
        Collects data from results dictionary and returns a DataFrame.
        Optionally allows passing custom names for both data keys and DataFrame columns.

        Parameters:
        - results: dict, the input dictionary containing the data.
        - custom_names: dict, optional, a dictionary mapping original keys to custom column names.

        Returns:
        - pd.DataFrame, the resulting DataFrame with the collected data.
        """

        # Default keys and their corresponding column names in the results dictionary
        if not custom_names:
            custom_names = {
                'times_elapsed': 'Times_Elapsed',
                'mse_odeint': 'MSE_odeint',
                'mse_coll_ode': 'MSE_collocation',
                'mse_odeint_test': 'MSE_odeint_test',
                'mse_coll_ode_test': 'MSE_collocation_test'
            }

        keys = list(set(k[0] for k in results.keys()))

        # Function to initialize a dictionary with the unique keys
        def init_di():
            return {key: [] for key in keys}

        # Initialize the data dictionary dynamically using the custom column names
        data_dict = {custom_name: init_di() for custom_name in custom_names.values()}

        # Populate the data dictionary with values from results
        for key in keys:
            for k, v in results.items():
                if k[0] == key:
                    for original_key, column_name in custom_names.items():
                        data_dict[column_name][key].append(v[original_key])

        # Create the DataFrame using the populated data dictionary
        df = pd.DataFrame(data_dict, index=keys)
        df.sort_index(inplace=True)

        return df
            
    @staticmethod
    def collect_data_toy(results):
        flattened_data = []

        for key, metrics in results.items():
            entry = {}
            if not isinstance(key, tuple):
                key = [key]
            for i, param in enumerate(key, start=1):
                entry[f'param{i}'] = param

            entry.update(metrics)

            flattened_data.append(entry)

        df = pd.DataFrame(flattened_data)
        return df    
    
    @staticmethod
    def series_to_lists(col):
        """
        Stacks columns of a DataFrame into lists.
        To be used for plotting (boxplots, etc.)
        """
        l = [i.item() for i in col]
        return l
    
    @staticmethod
    def columns_to_lists(df):
        """
        Stacks columns of a DataFrame into lists.
        To be used for plotting (boxplots, etc.)
        """
        rows_as_lists = {col: df[col].tolist() for col in df}
        return rows_as_lists
    
    @staticmethod
    def prep_for_boxplots(df, col_x, col_y):
        df_grouped = Results.group_into_lists(df, col_x, col_y)
        if not isinstance(df_grouped, pd.DataFrame):
            df_grouped = pd.DataFrame(df_grouped)
        df_box_plot = Results.columns_to_lists(df_grouped)
        df_box_plot.update({'x_labels': df_grouped.index.tolist()})
        return pd.DataFrame(df_box_plot)
    
    @staticmethod
    def filter_by_labels(source_df, reference_df, label_column='x_labels'):
        """
        Filters rows of source_df to only those where the label_column values are in reference_df.

        """
        source_df = source_df.copy()
        if label_column not in source_df.columns or label_column not in reference_df.columns:
            raise ValueError(f"The specified label_column '{label_column}' must exist in both DataFrames.")

        # convert list to tuple for hashable type in the label_column
        source_df['label_copy'] = source_df[label_column].apply(tuple)
        reference_df['label_copy'] = reference_df[label_column].apply(tuple)

        # create a set of labels from reference_df for fast lookup
        labels_set = set(reference_df['label_copy'])

        # filter source_df where label_column values are in the set from reference_df
        filtered_df = source_df[source_df['label_copy'].isin(labels_set)]
        filtered_df.drop(columns=['label_copy'], inplace=True)

        return filtered_df
    
def has_nested_tuple(t):
    for item in t:
        if isinstance(item, tuple):
            return True
    return False

def reload_module(module_name, class_name):
    module = importlib.import_module(module_name)
    importlib.reload(module)
    return getattr(module, class_name)

def convert_lists_in_tuple(param_tuple):
    """
    Converts all list elements in a tuple to string representations,
    keeping all other elements unchanged.
    """
    
    return tuple(str(item) if isinstance(item, list) else item for item in param_tuple)