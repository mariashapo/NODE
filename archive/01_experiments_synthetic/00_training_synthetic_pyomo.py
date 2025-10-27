{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import jax\n",
    "import jax.numpy as jnp\n",
    "\n",
    "import sys\n",
    "import os\n",
    "import pickle\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import importlib\n",
    "\n",
    "def append_path(path):\n",
    "    if path not in sys.path:\n",
    "        sys.path.append(path)\n",
    "        \n",
    "append_path(os.path.abspath(os.path.join('..', 'utils_training')))\n",
    "\n",
    "import optimize_pyomo_synthetic\n",
    "from optimize_pyomo_synthetic import reload_and_get_attribute\n",
    "\n",
    "Runner = reload_and_get_attribute(optimize_pyomo_synthetic, 'ExperimentRunner')\n",
    "runner = Runner('config.json')\n",
    "runner.run('default')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "dict_keys(['model_params', 'data', 'optimization_types'])\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/mariiashapo/Library/CloudStorage/OneDrive-Personal/project_2324/NODE/utils/non_parametric_collocation.py:95: UserWarning: Data transposed to match expected dimensions.\n",
      "  warnings.warn(\"Data transposed to match expected dimensions.\")\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "current_16_08\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/mariiashapo/Library/CloudStorage/OneDrive-Personal/project_2324/NODE/models/nn_pyomo_base.py:86: UserWarning: y_init should be structured such that each row represents a new time point.\n",
      "  warnings.warn(\"y_init should be structured such that each row represents a new time point.\")\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Ipopt 3.14.16: tol=0.0001\n",
      "print_level=1\n",
      "max_iter=3000\n",
      "\n",
      "\n",
      "******************************************************************************\n",
      "This program contains Ipopt, a library for large-scale nonlinear optimization.\n",
      " Ipopt is released as open source code under the Eclipse Public License (EPL).\n",
      "         For more information visit https://github.com/coin-or/Ipopt\n",
      "******************************************************************************\n",
      "\n",
      "{'solver_time': 2.0615270137786865, 'termination_condition': <TerminationCondition.optimal: 'optimal'>, 'message': 'Ipopt 3.14.16\\\\x3a Optimal Solution Found'}\n",
      "Iteration: 1 / 1\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "({None: {'time_elapsed': 2.0615270137786865,\n",
       "   'mse_train': Array(0.00577944, dtype=float64),\n",
       "   'mse_test': Array(0.0034757, dtype=float64),\n",
       "   'termination': <TerminationCondition.optimal: 'optimal'>}},\n",
       " <run_train_toy.TrainerToy at 0x28a5ca1c0>)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Runner = reload_and_get_attribute(optimize_pyomo_synthetic, 'ExperimentRunner')\n",
    "runner = Runner('config.json')\n",
    "runner.run('default')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "collocation_env",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.18"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
