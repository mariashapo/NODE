import numpy as np
import jax
import jax.numpy as jnp

import sys
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import importlib

def append_path(path):
    if path not in sys.path:
        sys.path.append(path)
        
append_path(os.path.abspath(os.path.join('..', '00_utils_training')))

import optimize_pyomo_synthetic
from optimize_pyomo_synthetic import reload_and_get_attribute

Runner = reload_and_get_attribute(optimize_pyomo_synthetic, 'ExperimentRunner')
runner = Runner('config.json')
results = runner.run('default')

print(results)