import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import importlib
import ast
import jax.numpy as jnp
import os
import pickle
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

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
    def plot_single_boxplot(
        data, labels, title, ylabel,
        color='blue', label='Data Label',
        x_label='Model Size Configuration', y_log=True,
        label_fontsize=14, tick_fontsize=12, title_fontsize=16,
    ):
        n_groups = len(data)
        positions = [i + 1 for i in range(n_groups)]
        
        plt.figure(figsize=(10, 6))
        box = plt.boxplot(data, positions=positions, widths=0.6, patch_artist=True, boxprops=dict(facecolor=color))
        
        plt.title(title, fontsize=title_fontsize)
        plt.xlabel(x_label, fontsize=label_fontsize)
        plt.ylabel(ylabel, fontsize=label_fontsize)
        if y_log:
            plt.yscale('log')
        
        plt.xticks(ticks=positions, labels=labels, fontsize=tick_fontsize)
        plt.tick_params(axis='y', labelsize=tick_fontsize)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_reg_curve_ci(
        x, y, y_lo, y_hi, *,
        title=None, xlabel=None, ylabel=None,
        xscale="log", yscale="linear",
        marker="o", linewidth=1.5, alpha_band=0.20,
        show_points=True, add_errorbars=False,
        y_min_clip=None, y_max=None, ax=None,
        title_on=True,
        preserve_label_case=False,
        label_fontsize=14,
        title_fontsize=16,
        inset=False,
        inset_min_x=1e-2,
        inset_max_x=None,
        inset_loc="lower left",
        tick_fontsize=None,
    ):
        """
        Regularization curve with shaded confidence interval band (#1).

        Parameters
        ----------
        x : array-like
            Regularization strengths (typically log-spaced).
        y : array-like
            Mean metric at each x.
        y_lo, y_hi : array-like
            Lower/upper CI bounds at each x (same length as y).
        xscale, yscale : str
            Axis scales (default: log x-axis).
        alpha_band : float
            Transparency for the CI band.
        y_min_clip, y_max : float, optional
            If provided, set y-limits and clip CI band to y_min_clip to avoid log underflow.
            If None and yscale='log', y_min_clip defaults to 0.5 * min(y[y>0]) and
            y_max defaults to 1.2 * max(y_hi).
        add_errorbars : bool
            Overlay symmetric error bars derived from y_lo/y_hi.
        ax : matplotlib axis, optional
            Plot onto an existing axis; if None, create a new figure/axis.
        title_on : bool
            If False, suppress the title even if provided.
        preserve_label_case : bool
            If True, use xlabel/ylabel as-is (no underscore/title casing).
        inset : bool
            If True, draw a zoomed inset for x >= inset_min_x (and <= inset_max_x if provided).
        inset_min_x, inset_max_x : float
            X-range for inset (log scale assumed); inset_max_x defaults to no upper bound.
        inset_loc : str
            Location string for inset_axes (e.g., 'lower left').
        """

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        y_lo = np.asarray(y_lo, dtype=float)
        y_hi = np.asarray(y_hi, dtype=float)

        # Sort by x so the line/band render correctly for log-spaced grids
        order = np.argsort(x)
        x, y, y_lo, y_hi = x[order], y[order], y_lo[order], y_hi[order]

        created_new_ax = ax is None
        ax = ax or plt.figure(figsize=(8, 5)).gca()

        # For log scale, enforce multiplicative (symmetric in log space) bands
        if yscale == "log":
            eps = 1e-16
            rel_lower = np.divide(y, y_lo, out=np.ones_like(y), where=(y_lo > eps))
            rel_upper = np.divide(y_hi, y, out=np.ones_like(y), where=(y > eps))
            rel = np.maximum(rel_lower, rel_upper)
            rel[rel <= 0] = 1.0
            y_lo_sym = y / rel
            y_hi_sym = y * rel
        else:
            y_lo_sym = np.copy(y_lo)
            y_hi_sym = np.copy(y_hi)

        # Default clipping for log scale to avoid CI hitting zero
        if y_min_clip is None and yscale == "log":
            positive_vals = np.concatenate([y[y > 0], y_lo_sym[y_lo_sym > 0], y_hi_sym[y_hi_sym > 0]])
            if positive_vals.size:
                y_min_clip = 0.5 * positive_vals.min()
        if y_max is None and yscale == "log":
            y_max = 1.2 * np.nanmax(y_hi_sym)

        y_lo_fill = np.copy(y_lo_sym)
        y_hi_plot = np.copy(y_hi_sym)
        if y_min_clip is not None:
            y_lo_fill = np.maximum(y_lo_fill, y_min_clip)

        # Mean curve
        ax.plot(x, y, marker=marker if show_points else None, linewidth=linewidth)

        # Shaded CI band
        ax.fill_between(x, y_lo_fill, y_hi_plot, alpha=alpha_band)

        # Optional error bars for visibility of small intervals
        if add_errorbars:
            if yscale == "log":
                yerr = np.vstack([y - y_lo_sym, y_hi_sym - y])
            else:
                yerr = np.vstack([y - y_lo, y_hi - y])
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="none",
                capsize=5,
                elinewidth=2,
                capthick=2,
                zorder=6,
                color=ax.lines[-1].get_color(),
            )

        if title and title_on:
            ax.set_title(title, fontsize=title_fontsize)
        if xlabel:
            ax.set_xlabel(xlabel if preserve_label_case else xlabel.replace("_", " ").title(), fontsize=label_fontsize)
        if ylabel:
            ax.set_ylabel(ylabel if preserve_label_case else ylabel.replace("_", " ").title(), fontsize=label_fontsize)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if tick_fontsize is not None:
            ax.tick_params(labelsize=tick_fontsize)
        if y_min_clip is not None or y_max is not None:
            ax.set_ylim(bottom=y_min_clip, top=y_max)

        ax.grid(True, which="major", linestyle="--", alpha=0.35)
        ax.grid(False, which="minor")
        if ax.figure:
            ax.figure.tight_layout()
        if created_new_ax:
            plt.show()

        # Optional inset for high-lambda region
        if inset:
            mask = x >= inset_min_x
            if inset_max_x is not None:
                mask &= x <= inset_max_x
            if np.any(mask):
                axins = inset_axes(ax, width="45%", height="45%", loc=inset_loc)
                axins.plot(x[mask], y[mask], marker=marker if show_points else None, linewidth=linewidth)
                axins.fill_between(x[mask], y_lo_fill[mask], y_hi_plot[mask], alpha=alpha_band)
                if add_errorbars:
                    if yscale == "log":
                        yerr_ins = np.vstack([y[mask] - y_lo_sym[mask], y_hi_sym[mask] - y[mask]])
                    else:
                        yerr_ins = np.vstack([y[mask] - y_lo[mask], y_hi[mask] - y[mask]])
                    axins.errorbar(
                        x[mask],
                        y[mask],
                        yerr=yerr_ins,
                        fmt="none",
                        capsize=5,
                        elinewidth=2,
                        capthick=2,
                        zorder=6,
                        color=ax.lines[-1].get_color(),
                    )
                axins.set_xscale(xscale)
                axins.set_yscale(yscale)
                axins.set_ylim(
                    bottom=0.8 * np.nanmin(y_lo_sym[mask]),
                    top=1.2 * np.nanmax(y_hi_sym[mask]),
                )
                axins.grid(True, which="major", linestyle="--", alpha=0.25)
                axins.grid(False, which="minor")
                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linewidth=0.8)


class GraphsTraining:
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
        Collects data from results dictionary and returns a DataFrame. Optionally allows passing custom names for both data keys and DataFrame columns.

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
    def collect_data_into_df(results, key_list = None):
        """Simplified method for synthetic data collection.
        
        Expected data structure {(param1, param2, param2) : metrics}, where metrics is also a dictionary.
        
        key_list (list) : optionally specify the column names for the key (hyperparameter) entries.  
        """
        flattened_data = []

        for key, metrics in results.items():
            entry = {}
            if not isinstance(key, tuple):
                key = [key]
            if key_list is not None and len(key_list) != len(key):
                # Skip entries whose key tuple does not match the expected length
                # to avoid breaking downstream aggregation.
                print(f"Skipping entry with key {key} (expected {len(key_list)} elements).")
                continue
            for i, param in enumerate(key):
                # each parameter from the key tuple is saved into the dataframe
                if key_list is None:
                    entry[f'param{i+1}'] = param # save as a generic parameter
                else:
                    entry[key_list[i]] = param

            # the dictionary values are then appended
            entry.update(metrics)

            # add each entry as a row
            flattened_data.append(entry)

        df = pd.DataFrame(flattened_data)
        
        numeric_cols = [col for col in df.columns if "mse" in col.lower() or "time" in col.lower()] # a bit of hardcoding in-here
        df[numeric_cols] = df[numeric_cols].astype("float")
        return df    
    
    @staticmethod
    def parse_sequential_training(one_seed_result):
        """Parse results coming from the sequential training."""
        # For older logs: time_elapsed was a list (possibly multiple stages).
        # For newer JAX timing-only runs: time_elapsed is a scalar float.
        t_info = one_seed_result.get("time_elapsed")
        pre_time = float(one_seed_result.get("pyomo_pretraining_time", 0.0) or 0.0)
        if isinstance(t_info, (list, tuple, np.ndarray)):
            total_elapsed = t_info[-1]
            t_before = np.sum(t_info[:-1]) if len(t_info) > 1 else 0.0
        else:
            total_elapsed = float(t_info)
            t_before = 0.0
        # Shift the timeline by any recorded Pyomo pretraining time so the
        # training curve starts after pretraining.
        t_before += pre_time

        train_loss_raw = one_seed_result["train_loss"]
        # train_loss is stored as [train_losses, test_losses]
        training_loss = np.array(train_loss_raw[-1][0])
        testing_loss = np.array(train_loss_raw[-1][1])

        n_records = len(training_loss)
        t_per_iter = total_elapsed / max(n_records, 1)
        
        iters = np.arange(1, n_records + 1)
        t_iters = iters * t_per_iter + t_before

        df = pd.DataFrame({"time_elapsed": t_iters, "mse_train": training_loss, "mse_test": testing_loss})
        df["system"] = one_seed_result.get("data_type")
        df["pretrain"] = bool(t_before > 0)
        df["max_iter"] = str(one_seed_result.get("max_iter"))
        
        return df
        
    @staticmethod
    def flatten_convergence_data(results_li : list, model_type : str, key_list = None) -> pd.DataFrame:
        """Flatten a list of convergence data dictionaries into a dataframe.
        
        model_type (str) : {"pyomo", "pytorch", "jax"}
        """
        dfs_li = []
        for i, result in enumerate(results_li):
            if model_type == "pyomo":
                df = Results.collect_data_into_df(result, key_list)
            else:
                df = Results.parse_sequential_training(result)
            if "seed" not in df.columns:
                df["seed"] = i
            dfs_li.append(df)
            
        full_df = pd.concat(dfs_li)
        return full_df
        
    
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
 
 
class ConvergenceCI:
    @staticmethod
    def _step_interp(x_src, y_src, x_grid, *, extrapolate_back=True):
        """
        Right-continuous step interpolation with optional backward-only extrapolation.

        - If extrapolate_back=True, x_grid < x_src[0] uses y_src[0].
        - We NEVER extrapolate forward; x_grid > x_src[-1] -> NaN.
        """
        idx = np.searchsorted(x_src, x_grid, side='right') - 1
        yg = np.where((idx >= 0) & (idx < y_src.size), y_src[np.clip(idx, 0, y_src.size-1)], np.nan)
        # no forward extrapolation
        yg = np.where(x_grid > x_src[-1], np.nan, yg)
        # backward-only extrapolation if requested
        if extrapolate_back:
            yg = np.where(x_grid < x_src[0], y_src[0], yg)
        else:
            yg = np.where(x_grid < x_src[0], np.nan, yg)
        return yg

    @staticmethod
    def time_ci(
        df, *,
        x_col='time_elapsed', y_col='mse_train', seed_col='seed',
        grid_points=200, tmax_quantile=0.9,
        interp='linear',
        extrapolate=True,  # backward-only extrapolation
        alpha=0.05, logspace=True, eps=1e-12,
        cutoff_missing_frac=0.5,  # new parameter
    ):
        curves, tmax_list, tmin_list = [], [], []
        for _, g in df.groupby(seed_col):
            x = np.asarray(g[x_col], dtype=float)
            y = np.asarray(g[y_col], dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            if x.size == 0:
                continue
            o = np.argsort(x, kind='mergesort')
            x, y = x[o], y[o]
            _, idx_rev = np.unique(x[::-1], return_index=True)
            idx = np.sort((x.size - 1) - idx_rev)
            x, y = x[idx], y[idx]
            curves.append((x, y))
            tmax_list.append(x[-1])
            tmin_list.append(x[0])
        if not curves:
            raise ValueError("No curves after grouping by seed.")

        tmax_arr = np.array(tmax_list, dtype=float)
        tmin_arr = np.array(tmin_list, dtype=float)

        Tq = float(np.quantile(tmax_arr, tmax_quantile))
        x_grid = np.linspace(0.0, Tq, int(grid_points))

        def _interp_linear(x, y, xg):
            left_val = y[0] if extrapolate else np.nan
            yg = np.interp(xg, x, y, left=left_val, right=np.nan)
            return yg

        def _interp_step(x, y, xg):
            idx = np.searchsorted(x, xg, side='right') - 1
            yg = np.where((idx >= 0) & (idx < y.size), y[np.clip(idx, 0, y.size-1)], np.nan)
            yg = np.where(xg > x[-1], np.nan, yg)  # no forward extrap
            if extrapolate:
                yg = np.where(xg < x[0], y[0], yg)
            return yg

        fn = _interp_linear if interp == 'linear' else _interp_step
        Y = np.vstack([fn(x, y, x_grid) for (x, y) in curves])  # (n_seeds, T)
        finite_mask = np.isfinite(Y)
        counts = np.sum(finite_mask, axis=0).astype(float)
        n_seeds = len(curves)
        avail_frac = counts / n_seeds

        min_support_abs = 3

        too_sparse_frac = avail_frac < (1 - cutoff_missing_frac)
        too_sparse_abs  = counts < min_support_abs
        too_sparse_mask = too_sparse_frac | too_sparse_abs

        # Mask out undersupported grid points
        if np.any(too_sparse_mask):
            Y[:, too_sparse_mask] = np.nan
            finite_mask[:, too_sparse_mask] = False
            counts[too_sparse_mask] = 0.0

        # (rest identical to before)
        x0 = tmin_arr[:, None]
        Xg = x_grid[None, :]
        pre_mask_matrix = (Xg < x0) & finite_mask
        pre_count = np.sum(pre_mask_matrix, axis=0).astype(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            pre_frac = pre_count / counts
        pre_mask = np.isfinite(pre_frac) & (pre_frac == 1.0)

        from scipy.stats import norm
        z = norm.ppf(1 - alpha/2.0)
        if logspace:
            Yp = np.where(Y > eps, Y, np.nan)
            L = np.log(Yp)
            mean_L = np.nanmean(L, axis=0)
            std_L  = np.nanstd(L, axis=0, ddof=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                half_L = z * std_L / np.sqrt(counts)
            mean = np.exp(mean_L)
            lo   = np.exp(mean_L - half_L)
            hi   = np.exp(mean_L + half_L)
        else:
            mean = np.nanmean(Y, axis=0)
            std  = np.nanstd(Y, axis=0, ddof=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                half = z * std / np.sqrt(counts)
            lo, hi = mean - half, mean + half

        meta = dict(
            counts=counts,
            avail_frac=avail_frac,
            too_sparse_mask=too_sparse_mask,
            pre_frac=pre_frac,
            tmin_global=float(np.min(tmin_arr)),
            tmax_quantile=Tq,
        )
        return x_grid, mean, lo, hi, Y, pre_mask, meta


def plot_convergence_debug(
    df, *,
    x_col='time_elapsed', y_col='mse_train', seed_col='seed',
    grid_points=200, tmax_quantile=0.95,
    interp='linear', extrapolate=False,
    ci=True, alpha_band=0.25, lw=1.8,
    logy=True, figsize=(10, 5),
    color_mean='C0', color_traces='gray', alpha_traces=0.4
):
    """
    Plot all per-seed convergence curves and (optionally) the aggregate CI.

    Parameters
    ----------
    df : DataFrame from Results.flatten_convergence_list
    system, pretrain : filters
    interp : 'linear' | 'step' | None
    extrapolate : extend curves past last x value when True
    ci : whether to overlay mean ± CI from time_ci()
    logy : log-scale for MSE
    """
    fig, ax = plt.subplots(figsize=figsize)

    # ---- 1. plot all seed traces (raw, not averaged)

    for s, g in df.groupby(seed_col):
        x = np.asarray(g[x_col], dtype=float)
        y = np.asarray(g[y_col], dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if len(x) == 0:
            continue
        order = np.argsort(x)
        x, y = x[order], y[order]
        ax.plot(x, y, color=color_traces, lw=1.0, alpha=alpha_traces)

    # ---- 2. optionally overlay mean + CI
    if ci:
        xg, mean, lo, hi, _ = ConvergenceCI.time_ci(
            df,
            x_col=x_col, y_col=y_col, seed_col=seed_col,
            grid_points=grid_points, tmax_quantile=tmax_quantile,
            interp=interp, extrapolate=extrapolate,
        )
        ax.plot(xg, mean, color=color_mean, lw=lw, label="Mean")
        ax.fill_between(xg, lo, hi, color=color_mean, alpha=alpha_band, label="95% CI")

    # ---- 3. style
    if logy:
        ax.set_yscale('log')
        ax.set_ylabel("Training MSE (log scale)")
    else:
        ax.set_ylabel("Training MSE")

    ax.set_xlabel("Training Time (s)")
    ax.grid(True, which='both', ls=':', lw=0.6, alpha=0.6)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    return ax

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


def plot_ci_fn(x, mean, lo, hi, label=None, logy=True, color=None, alpha_fill=0.2, linewidth=2.5):
    """
    Simple confidence interval plotting helper.
    Draws mean + shaded CI region on existing axes.
    """
    plt.plot(x, mean, label=label, linewidth=linewidth, color=color)
    plt.fill_between(x, lo, hi, alpha=alpha_fill, color=color)


def plot_time_bands(
    df_map, *,
    y_col='mse_train',
    grid_points=200,
    tmax_quantile=0.9,
    align_grid=True,
    line_width=2.0,
    band_alpha=0.25,
    logy=True,
    grid=True,
    title=None,
    extrapolate=True,  # backward-only semantics (matches time_ci)
    cutoff_missing_frac = 0.5,
    figsize=(12, 7),
    xlim=None,
    ylim=None,
    label_fontsize=None,
    tick_fontsize=None,
    legend_fontsize=None,
    title_fontsize=None
):
    """Plot mean/CI bands of convergence curves over training time.

    Args:
        df_map: Dict[label -> DataFrame] containing time/metric columns expected by ConvergenceCI.time_ci.
        y_col: Metric column to plot (e.g., 'mse_train' or 'mse_test').
        grid_points: Number of points in the interpolated time grid.
        tmax_quantile: Float or dict per-label quantile for truncating long tails.
        align_grid: If True, align all curves to a common time grid.
        line_width: Line width for mean curves.
        band_alpha: Fill opacity for confidence bands.
        logy: Plot y-axis on log scale.
        grid: Toggle background grid.
        title: Optional plot title.
        extrapolate: Allow backward extrapolation to cover pretrain regions.
        cutoff_missing_frac: Drop regions with excessive missing coverage.
        figsize: Figure size tuple passed to matplotlib.
        xlim: Optional (min, max) for x-axis.
        ylim: Optional (min, max) for y-axis.
        label_fontsize: Optional font size for x/y labels.
        tick_fontsize: Optional font size for tick labels.
        legend_fontsize: Optional font size for legend text.
        title_fontsize: Optional font size for the title.

    Returns:
        Matplotlib Axes with plotted bands and legend.
    """
    def _has_pretraining(df):
        """Detect pretraining from common boolean columns."""
        for col in ("pretrain", "pretraining", "pyomo_pretraining"):
            if col in df.columns and df[col].astype(bool).any():
                return True
        return False

    pretraining_present = any(_has_pretraining(df) for df in df_map.values())
    curves = {}

    # --- compute curves ---
    for label, df in df_map.items():
        if type(tmax_quantile) == dict:
            tmax_q = tmax_quantile[label]
        else:
            tmax_q = tmax_quantile
        x, mean, lo, hi, Y, pre_mask, meta = ConvergenceCI.time_ci(
            df,
            grid_points=grid_points,
            tmax_quantile=tmax_q,
            y_col=y_col,
            extrapolate=extrapolate,  # backward-only extrapolation
            logspace=logy,
            cutoff_missing_frac = cutoff_missing_frac
        )
        curves[label] = dict(
            x=x, mean=mean, lo=lo, hi=hi,
            pre_mask=pre_mask,
            t_pre_end=meta['tmin_global']  # handy if we realign grids
        )

    # --- optionally align x-grids across labels ---
    if align_grid:
        all_x = np.concatenate([c["x"] for c in curves.values()])
        x_common = np.linspace(np.nanmin(all_x), np.nanmax(all_x), grid_points)

        for d in curves.values():
            # Preserve backward-only extrapolation: left=first value, right=np.nan
            d["mean"] = np.interp(x_common, d["x"], d["mean"],
                                  left=d["mean"][0], right=np.nan)
            d["lo"]   = np.interp(x_common, d["x"], d["lo"],
                                  left=d["lo"][0],   right=np.nan)
            d["hi"]   = np.interp(x_common, d["x"], d["hi"],
                                  left=d["hi"][0],   right=np.nan)
            d["x"] = x_common
            # Recompute dashed mask from stored boundary
            d["pre_mask"] = x_common < d["t_pre_end"]

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)
    eps = 1e-12 if logy else 0.0

    any_pre = False
    for label, d in curves.items():
        x = d["x"]
        mean = np.clip(d["mean"], eps, None)
        lo   = np.clip(d["lo"],   eps, None)
        hi   = np.clip(d["hi"],   eps, None)
        pre  = d["pre_mask"]
        post = ~pre

        # Plot the non-pre-training (observed/support) part first to anchor color & legend
        # We allow this to be empty; matplotlib will still assign a color.
        [main_line] = ax.plot(x[post], mean[post], label=label, linewidth=line_width)
        color = main_line.get_color()

        # Fill band for the whole domain (NaNs create gaps automatically)
        ax.fill_between(x, lo, hi, alpha=band_alpha, facecolor=color, edgecolor='none')

        # Plot the pre-training (pure backward-extrapolated) part with dashed style
        if np.any(pre):
            any_pre = True
            ax.plot(x[pre], mean[pre], linewidth=line_width, linestyle='--', color=color)

    if logy:
        ax.set_yscale("log")

    ax.set_xlabel("Training Time (s)")
    ylabel = "Training MSE" if y_col is None else y_col.replace("_", " ").title()
    ax.set_ylabel(ylabel + (" (log scale)" if logy else ""))

    if label_fontsize is not None:
        ax.xaxis.label.set_size(label_fontsize)
        ax.yaxis.label.set_size(label_fontsize)
    if tick_fontsize is not None:
        ax.tick_params(labelsize=tick_fontsize)

    if grid:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)

    if title:
        ax.set_title(title, fontsize=title_fontsize)

    # Optional axis overrides to "zoom out" or focus a region.
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Build legend with an extra entry for the dashed meaning.
    if any_pre and pretraining_present:
        pre_proxy = Line2D([0], [0], linestyle='--', color='black', label='pre-training')
        handles, labels = ax.get_legend_handles_labels()
        handles.append(pre_proxy)
        labels.append('pre-training')
        ax.legend(handles, labels, frameon=False, fontsize=legend_fontsize)
    else:
        ax.legend(frameon=False, fontsize=legend_fontsize)

    plt.tight_layout()
    plt.show()
    return ax


def load_all_pickles(folder, recursive=True):
    """Load all .pkl files from a folder into a list."""
    all_results = []
    for root, dirs, files in os.walk(folder):
        for fname in sorted(files):
            if fname.endswith(".pkl"):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "rb") as f:
                        data = pickle.load(f)
                    all_results.append(data)
                    print(f"Loaded {fpath}")
                except Exception as e:
                    print(f"⚠️ Skipping {fpath}: {e}")
        if not recursive:
            break
    print(f"\nLoaded {len(all_results)} files from {folder}")
    return all_results
