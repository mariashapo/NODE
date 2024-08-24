import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import importlib
import ast

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
                            x_label = 'Model Size Configuration'):
        n_groups = len(data)
        positions = [i + 1 for i in range(n_groups)]
        
        plt.figure(figsize=(10, 6))
        box = plt.boxplot(data, positions=positions, widths=0.6, patch_artist=True, boxprops=dict(facecolor=color))
        
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(ylabel)
        plt.yscale('log')
        
        plt.xticks(ticks=positions, labels=labels)
        
        patch = mpatches.Patch(color=color, label=label)  
        
        # plt.axhline(y=min_acc, color='r', linestyle='--', label='Minimum MSE')
        plt.legend(handles=[patch], loc='upper left')
        plt.grid(True)
        plt.show()


class Results:
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
    def group_into_lists(df, group_col, value_col):
        """
        Groups a DataFrame into lists.
        """
        grouped = df.groupby(group_col).agg(list)[value_col]
        return grouped
    
    @staticmethod
    def collect_data(results):
        """
        Collects data from results dictionary and returns a DataFrame.
        Keys are set as index.
        Values are lists of values for each key.
        """
        keys = [k[0] for k in results.keys()]
        keys = list(set(keys))

        def init_di():
            return {key: [] for key in keys}

        times_elapsed = init_di()
        mse_odeint = init_di()
        mse_coll_ode = init_di()
        mse_odeint_test = init_di()
        mse_coll_ode_test = init_di()

        for key in keys:
            for k, v in results.items():
                if k[0] == key:
                    times_elapsed[key].append(v['times_elapsed'])
                    mse_odeint[key].append(v['mse_odeint'])
                    mse_coll_ode[key].append(v['mse_coll_ode'])
                    mse_odeint_test[key].append(v['mse_odeint_test'])
                    mse_coll_ode_test[key].append(v['mse_coll_ode_test'])

        data = {
            'Times_Elapsed': times_elapsed,
            'MSE_odeint': mse_odeint,
            'MSE_collocation': mse_coll_ode,
            'MSE_odeint_test': mse_odeint_test,
            'MSE_collocation_test': mse_coll_ode_test
        }
        
        df = pd.DataFrame(data, index=keys)
        df.sort_index(inplace=True)
        return df
    
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