"""Quick viewer for Pyomo pretrain bundles (prints actual weight values)."""
import argparse
import pickle
from pathlib import Path
import numpy as np


def _shape(arr):
    try:
        return tuple(arr.shape)
    except Exception:
        return None


def _to_np(arr):
    try:
        return np.array(arr)
    except Exception:
        return arr


def _print_jax(weights_jax):
    if not weights_jax:
        print("  [JAX] no weights found")
        return
    print("  [JAX] layers (kernel, bias):")
    for name, params in weights_jax.items():
        ker = _to_np(params.get("kernel"))
        bias = _to_np(params.get("bias"))
        print(f"    - {name}: kernel{_shape(ker)} =\n{ker}")
        print(f"      bias{_shape(bias)} = {bias}")


def _print_pt(weights_pt):
    if not weights_pt:
        print("  [PyTorch] no weights found")
        return
    print("  [PyTorch] layers (W, b):")
    for i, (w, b) in enumerate(weights_pt):
        w_np = _to_np(w)
        b_np = _to_np(b)
        print(f"    - layer {i}: W{_shape(w_np)} =\n{w_np}")
        print(f"      b{_shape(b_np)} = {b_np}")


def _print_pyomo_raw(raw):
    if not raw:
        print("  [Pyomo raw] no weights found")
        return
    print("  [Pyomo raw] variables:")
    for k, v in raw.items():
        v_np = _to_np(v)
        print(f"    - {k}: shape{_shape(v_np)} =\n{v_np}")


def main():
    ap = argparse.ArgumentParser(description="Preview weights in a Pyomo pretrain bundle.")
    ap.add_argument("bundle", type=Path, help="Path to pyomo_pretrain_*.pkl")
    ap.add_argument("--suppress", action="store_true", help="Suppress scientific notation.")
    args = ap.parse_args()

    if args.suppress:
        np.set_printoptions(suppress=True, linewidth=160)
    else:
        np.set_printoptions(linewidth=160)

    with open(args.bundle, "rb") as f:
        bundle = pickle.load(f)

    print(f"Bundle: {args.bundle}")
    timing = bundle.get("timing", {})
    print(f"Timing: wall={timing.get('wall_time')}s solver={timing.get('solver_time')}s")
    metrics = bundle.get("metrics", {})
    if metrics:
        print(f"Metrics: mse_train={metrics.get('mse_train')}, mse_test={metrics.get('mse_test')}")

    _print_jax(bundle.get("weights_jax"))
    _print_pt(bundle.get("weights_pytorch"))
    _print_pyomo_raw(bundle.get("weights_pyomo_raw"))


if __name__ == "__main__":
    main()
