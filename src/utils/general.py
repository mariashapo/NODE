"""General utils."""
import os, time, psutil
import numpy as np
import time

def generate_seeds(n_seeds: int, method: str = "random"):
    """
    Generate a list of unique NumPy seeds.

    Parameters
    ----------
    n_seeds : int
        Number of seeds to generate.
    method : str, optional
        Method to generate seeds. Options:
        - 'random': uses np.random.SeedSequence for reproducibility
        - 'time': mixes in current time for unique results

    Returns
    -------
    list of int
        List of unique integer seeds.
    """
    if method == "random":
        # Use a reproducible random seed sequence
        seed_seq = np.random.SeedSequence()
        seeds = seed_seq.spawn(n_seeds)
        return [int(s.generate_state(1)[0]) for s in seeds]

    elif method == "time":
        # Use time-based seed generation for uniqueness across runs
        base = int(time.time() * 1e6)  # microsecond precision
        return [base + i for i in range(n_seeds)]

    else:
        raise ValueError("Method must be 'random' or 'time'.")
    
def print_memory(prefix=""):
    """Print current RSS (resident set size) in MB."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    print(f"{prefix}Memory usage: {mem_mb:.2f} MB")