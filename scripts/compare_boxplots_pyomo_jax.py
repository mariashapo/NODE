"""
Compare Pyomo vs JAX metrics via side-by-side boxplots (aligned by layer width).

Example:
  python scripts/compare_boxplots_pyomo_jax.py \
    --pyomo_dir results/study_vdp/pyomo_vdp_32_301225 \
    --jax_dir results/jax_pretrain_pyomo_vdp_w2-64-2/jax_vdp_5000_311225 \
    --metric mse_test \
    --min_runs 3 \
    --out results/plots/pyomo_vs_jax_width.png
"""

import argparse
import pickle
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.synthetic.aggregate_pyomo_reg_search import load_reg_search, aggregate_by_hparams, _format_layer_width
from utils.analyse_results import Graphs


def load_jax_folder(folder: Path):
    """Load JAX pickles (dict-of-dicts keyed by layer_widths/penalty/tol)."""
    vals = []
    for fp in sorted(folder.glob("*.pkl")):
        try:
            obj = pickle.load(fp.open("rb"))
        except Exception:
            continue
        vals.append(obj)
    return vals


def load_jax_meta(folder: Path):
    meta_path = folder / "run_meta.json"
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r") as f:
            return json.load(f)
    except Exception:
        return {}


def flatten_dict_of_dicts(objs, metric, default_lw=None):
    rows = []
    for obj in objs:
        if isinstance(obj, dict):
            # dict-of-dicts keyed by (lw, reg, tol)
            if all(isinstance(k, (tuple, list)) for k in obj.keys()):
                for k, v in obj.items():
                    if isinstance(k, (tuple, list)) and len(k) >= 3:
                        lw, reg, tol = k[:3]
                    else:
                        lw = default_lw
                        reg = tol = None
                    rows.append(
                        {
                            "layer_widths": tuple(lw) if isinstance(lw, (list, tuple)) else lw,
                            "penalty_lambda_reg": reg,
                            "tol": tol,
                            metric: v.get(metric, np.nan) if isinstance(v, dict) else np.nan,
                        }
                    )
            else:
                # flat dict with metrics
                lw = obj.get("layer_widths") or obj.get("layer_width") or default_lw
                reg = obj.get("penalty_lambda_reg")
                tol = obj.get("tol")
                rows.append(
                    {
                        "layer_widths": tuple(lw) if isinstance(lw, (list, tuple)) else lw,
                        "penalty_lambda_reg": reg,
                        "tol": tol,
                        metric: obj.get(metric, np.nan),
                    }
                )
        else:
            # unsupported type; skip
            continue
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare Pyomo vs JAX via boxplots (aligned by layer width).")
    ap.add_argument("--pyomo_dir", required=True, type=Path, help="Folder with Pyomo .pkl results.")
    ap.add_argument("--jax_dir", required=True, type=Path, help="Folder with JAX .pkl results.")
    ap.add_argument("--metric", default="mse_test", help="Metric to compare (e.g., mse_test, mse_train).")
    ap.add_argument("--min_runs", type=int, default=3, help="Minimum runs per group to include.")
    ap.add_argument("--out", type=Path, default=None, help="Optional path to save the figure (png).")
    args = ap.parse_args(argv)

    # Pyomo aggregate
    df_pyomo = load_reg_search(str(args.pyomo_dir))
    agg_pyomo = aggregate_by_hparams(df_pyomo)
    agg_pyomo = agg_pyomo[(agg_pyomo["n_runs"] >= args.min_runs)]

    # JAX flat load
    meta = load_jax_meta(args.jax_dir)
    default_lw = None
    if isinstance(meta, dict):
        default_lw = meta.get("layer_width") or meta.get("args", {}).get("layer_width")
    jax_objs = load_jax_folder(args.jax_dir)
    jax_rows = flatten_dict_of_dicts(jax_objs, args.metric, default_lw=default_lw)
    import pandas as pd

    df_jax = pd.DataFrame(jax_rows)
    agg_jax = (
        df_jax.groupby(["layer_widths"])
        .agg(
            metric_mean=(args.metric, "mean"),
            n_runs=("layer_widths", "count"),
        )
        .reset_index()
    )
    agg_jax = agg_jax[agg_jax["n_runs"] >= args.min_runs]

    # Align on layer_widths present in both
    common = set(agg_pyomo["layer_widths"]).intersection(set(agg_jax["layer_widths"]))
    agg_pyomo = agg_pyomo[agg_pyomo["layer_widths"].isin(common)]
    agg_jax = agg_jax[agg_jax["layer_widths"].isin(common)]
    if agg_pyomo.empty or agg_jax.empty:
        raise SystemExit("No overlapping layer_widths with sufficient runs.")

    # Build data lists aligned by sorted widths
    widths_sorted = sorted(common, key=lambda w: (_format_layer_width(w), w))
    labels = [_format_layer_width(w) for w in widths_sorted]

    pyomo_vals = []
    jax_vals = []
    for w in widths_sorted:
        py_vals = df_pyomo[df_pyomo["layer_widths"] == w][args.metric].dropna().values
        j_vals = df_jax[df_jax["layer_widths"] == w][args.metric].dropna().values
        if py_vals.size >= args.min_runs and j_vals.size >= args.min_runs:
            pyomo_vals.append(py_vals)
            jax_vals.append(j_vals)
        else:
            labels = [lbl for lbl, lw in zip(labels, widths_sorted) if lw != w]

    if not pyomo_vals or not jax_vals:
        raise SystemExit("No data left to plot after filtering min_runs.")

    plt.figure(figsize=(10, 6))
    Graphs.plot_boxplots(
        pyomo_vals,
        jax_vals,
        labels,
        title=f"{args.metric} Pyomo vs JAX",
        ylabel=args.metric.replace("_", " ").title(),
        colors=("C0", "C1"),
        color_labels=["Pyomo", "JAX"],
        x_label="Layer width",
        y_log=not args.metric.startswith("time"),
    )

    if args.out:
        plt.savefig(args.out, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
