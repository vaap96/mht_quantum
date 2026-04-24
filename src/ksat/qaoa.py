import numpy as np
import scipy.optimize as optimize
from tqdm import tqdm


def get_lr_params(deltas, p):
    """Expands 2 parameters (dg, db) into 2p parameters via linear ramp."""

    dg, db = deltas
    k = np.arange(1, p + 1)
    gammas = dg * (k / p)
    betas = db * (1.0 - k / p)

    return np.concatenate([gammas, betas])


def simulate_qaoa_fast(params, n, h_diag, p):
    """Fast Numpy Simulator for QAOA Statevector."""

    num_states = 2 ** n
    gammas = params[:p]
    betas = params[p:]
    psi = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    indices = np.arange(num_states)

    for t in range(p):
        psi *= np.exp(-1j * gammas[t] * h_diag)
        beta = betas[t]
        c = np.cos(beta)
        s = -1j * np.sin(beta)
        for qubit in range(n):
            mask_0 = (indices >> qubit) & 1 == 0
            idx0 = indices[mask_0]
            idx1 = idx0 + (1 << qubit)
            u0 = psi[idx0]
            u1 = psi[idx1]
            psi[idx0] = c * u0 + s * u1
            psi[idx1] = s * u0 + c * u1

    return np.abs(psi) ** 2


def train_full_model(training_data, train_n, p_depth):
    print("\n--- Training Full QAOA (Small Angle Init) ---")

    def objective(pars):
        total_p = 0.0
        for h_diag, sol_indices in training_data:
            probs = simulate_qaoa_fast(pars, train_n, h_diag, p_depth)
            total_p += np.sum(probs[sol_indices])
        return -1.0 * (total_p / len(training_data))

    # Paper Strategy: Initialize close to zero
    x0 = np.concatenate([np.full(p_depth, -0.01), np.full(p_depth, 0.01)])

    res = optimize.minimize(objective, x0, method='COBYLA', tol=1e-3, options={'maxiter': 600})
    print(f"  > Full QAOA Train Avg Prob: {-res.fun:.4f}")

    return res.x


def train_lr_grid_search(training_data, depth, train_n, skip_grid=False):
    print("\n--- Training LR QAOA (Grid Search Init) ---")

    def get_avg_p_succ(dg, db):
        pars = get_lr_params([dg, db], depth)
        total_p = 0.0
        for h_diag, sol_indices in training_data:
            probs = simulate_qaoa_fast(pars, train_n, h_diag, depth)
            total_p += np.sum(probs[sol_indices])
        return total_p / len(training_data)

    # 1. Grid Search
    # Scanning widely to find the correct basin of attraction
    dg_vals = np.linspace(-2.0, 2.0, 11)
    db_vals = np.linspace(0.1, 4.0, 11)

    best_p = -1.0
    best_deltas = [-0.8, 0.49]

    if not skip_grid:
        for dg in tqdm(dg_vals, desc="Grid Scanning"):
            for db in db_vals:
                p_val = get_avg_p_succ(dg, db)
                if p_val > best_p:
                    best_p = p_val
                    best_deltas = [dg, db]

    print(f"  > Best Grid Start: dg={best_deltas[0]:.2f}, db={best_deltas[1]:.2f} (Prob={best_p:.4f})")

    # 2. Refinement
    def objective(d):
        return -1.0 * get_avg_p_succ(d[0], d[1])

    res = optimize.minimize(objective, best_deltas, method='COBYLA', tol=1e-3, options={'maxiter': 200})
    print(f"  > LR QAOA Train Avg Prob: {-res.fun:.4f}")

    return get_lr_params(res.x, depth)
