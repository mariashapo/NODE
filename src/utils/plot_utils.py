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

    return ax
