import numpy as np
import matplotlib.pyplot as plt

def plot_convergence_ci(
    x, mean, lo, hi,
    label="Pyomo",
    ax=None,
    logy=True,
    band_alpha=0.25,
    line_width=2.0,
    grid=True,
    pretrain_time=None,   # float or None
    pretrain_marker="x",
    pretrain_size=70,
):
    """
    Plot a single convergence curve with confidence band.

    Parameters
    ----------
    x, mean, lo, hi : 1D arrays
        Output of your CI computation (e.g., from ConvergenceCI.time_ci).
    label : str
        Legend label for the curve.
    ax : matplotlib Axes or None
        If None, uses current axes.
    logy : bool
        Whether to use log scale on the y-axis.
    band_alpha : float
        Transparency for the confidence band.
    line_width : float
        Line width for the mean curve.
    grid : bool
        Whether to draw a light grid.
    pretrain_time : float or None
        If provided, marks a vertical line / point at the end of pre-training.
    pretrain_marker : str
        Marker used at the pre-training time (if provided).
    pretrain_size : int
        Marker size for the pre-training marker.
    """
    ax = ax if ax is not None else plt.gca()

    # mean + band
    ax.plot(x, mean, label=label, linewidth=line_width)
    ax.fill_between(x, lo, hi, alpha=band_alpha)

    # axis styling
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Training Time (s)")
    ax.set_ylabel("Training MSE" + (" (log scale)" if logy else ""))

    if grid:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)

    # optional pre-training marker
    if pretrain_time is not None and np.isfinite(pretrain_time):
        # pick the y at (or just before) that time for a marker
        idx = np.searchsorted(x, pretrain_time, side="right") - 1
        idx = max(0, min(idx, len(x)-1))
        ax.scatter([x[idx]], [mean[idx]], marker=pretrain_marker, s=pretrain_size, zorder=5)
        ax.axvline(pretrain_time, color=ax.lines[-1].get_color(), linestyle=":", linewidth=1)

    return ax
