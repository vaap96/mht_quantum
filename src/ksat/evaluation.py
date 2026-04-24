import numpy as np
from tqdm import tqdm

from src.ksat.qaoa import simulate_qaoa_fast
from src.ksat.walksat import fast_walksat_solver, walksatlm_paper_kernel


def evaluate_model(dataset, params_lr, p_depth):
    print(f"\n--- Comparative Evaluation (LR Params) ---")
    results = {'n': [], 'lr_runtime': []}

    # Sort keys to ensure sequential processing
    n_values = sorted(dataset.keys())

    for n in n_values:
        instances = dataset[n]
        runtimes_lr = []

        # Iterate through the pre-generated instances
        for instance in tqdm(instances, desc=f"Benchmarking n={n}", leave=False):
            h_diag = instance['h_diag']
            sol_indices = instance['sol_indices']

            # Simulate QAOA using the provided Linear Response parameters
            probs_lr = simulate_qaoa_fast(params_lr, n, h_diag, p_depth)

            # Calculate overlap with solution subspace
            # Added epsilon 1e-15 to prevent division by zero in runtime calculation
            p_lr = max(np.sum(probs_lr[sol_indices]), 1e-15)

            # Metric: Expected number of shots to observe a solution (1/p)
            runtimes_lr.append(1.0 / p_lr)

        med_lr = np.median(runtimes_lr)

        results['n'].append(n)
        results['lr_runtime'].append(med_lr)

        tqdm.write(f"n={n}: Median Runtime (1/p) = {med_lr:.2f}")

    return results


def evaluate_model_walksat_fast(dataset, max_flips=100000, p_noise=0.5):
    """
    High-performance evaluation using Numba.
    Converts data structures to numpy matrices before solving.
    """

    print(f"\n--- Comparative Evaluation: WalkSAT (Numba Accelerated) ---")
    results = {'n': [], 'median_runtime': []}

    n_values = sorted(dataset.keys())

    for n in n_values:
        instances = dataset[n]
        runtimes = []

        for instance in tqdm(instances, desc=f"Benchmarking n={n}", leave=False):
            # 1. Convert specific tuple-list format to rigid Numpy Matrices for Numba
            # Structure: List of (vars_idx_array, signs_array)
            clauses_raw = instance['clauses']
            m = len(clauses_raw)

            # Allocate (M, 8) matrices
            # We assume K=8 based on your problem
            c_vars = np.zeros((m, 8), dtype=np.int32)
            c_signs = np.zeros((m, 8), dtype=np.int32)

            for i, (v_idx, signs) in enumerate(clauses_raw):
                c_vars[i, :] = v_idx
                c_signs[i, :] = signs

            # 2. Run Numba Solver
            flips = fast_walksat_solver(n, c_vars, c_signs, max_flips, p_noise)
            runtimes.append(flips)

        med_val = np.median(runtimes)
        results['n'].append(n)
        results['median_runtime'].append(med_val)

        tqdm.write(f"n={n}: Median Runtime = {med_val:.2f}")

    return results


def evaluate_model_walksatlm(dataset, max_flips=100000, p_noise=0.5, w1=6, w2=5):
    """
    Evaluates the Paper-Correct WalkSATlm on the dataset.
    """

    print(f"\n--- Comparative Evaluation: WalkSATlm (Paper Version, w1={w1}, w2={w2}) ---")
    results = {'n': [], 'median_runtime': []}

    n_values = sorted(dataset.keys())

    for n in n_values:
        instances = dataset[n]
        runtimes = []

        for instance in tqdm(instances, desc=f"Benchmarking n={n}", leave=False):
            # Prepare data for Numba
            clauses_raw = instance['clauses']
            m = len(clauses_raw)
            c_vars = np.zeros((m, 8), dtype=np.int32)
            c_signs = np.zeros((m, 8), dtype=np.int32)

            for i, (v_idx, signs) in enumerate(clauses_raw):
                c_vars[i, :] = v_idx
                c_signs[i, :] = signs

            # Run Solver
            flips = walksatlm_paper_kernel(n, c_vars, c_signs, max_flips, p_noise, w1, w2)
            runtimes.append(flips)

        med_val = np.median(runtimes)
        results['n'].append(n)
        results['median_runtime'].append(med_val)

        tqdm.write(f"n={n}: Median Runtime = {med_val:.2f}")

    return results


def fit_scaling(n, y):
    log_y = np.log(y)
    slope, intercept = np.polyfit(n, log_y, 1)

    return np.exp(intercept), slope
