import numpy as np
import scipy.optimize as optimize
from tqdm import tqdm


def get_lr_params(deltas, p):
    """
    Expands 2 parameters (dg, db) into 2p parameters via linear ramp.
    This is done based on the method described in: https://www.nature.com/articles/s41534-025-01082-1.pdf
    """

    dg, db = deltas
    k = np.arange(1, p + 1)
    gammas = dg * (k / p)
    betas = db * (1.0 - k / p)

    return np.concatenate([gammas, betas])


def simulate_qaoa_fast(params, n, h_diag, p):
    """
    Fast Numpy Simulator for QAOA Statevector (limit in the max number of qubits).
    """

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
    """
    Implements a standard QAOA procedure/training using variational training of 2p params.
    Returns:
        res.x (np.array): the optimal values found from the training procedure for the trainable params.
    """
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


def train_lr_qaoa_unified(h_diags, depth, train_n, sol_indices_list=None, skip_grid=False):
    """
    Unified LR-QAOA optimizer.
    - If sol_indices_list is provided, it maximizes success probability (Supervised).
    - If sol_indices_list is None, it minimizes expected energy (Unsupervised).
    """
    is_supervised = sol_indices_list is not None
    mode_name = "Probability (Supervised)" if is_supervised else "Expected Energy (Unsupervised)"
    print(f"\n--- Training LR QAOA: {mode_name} ---")

    # PRE-PROCESSING: Create a normalized version of the Hamiltonians
    norm_h_diags = []
    for h in h_diags:
        max_val = np.max(np.abs(h))
        # Avoid dividing by zero if the Hamiltonian is empty/flat
        scale_factor = max_val if max_val > 0 else 1.0
        norm_h_diags.append(h / scale_factor)

    def get_objective_value(dg, db):
        pars = get_lr_params([dg, db], depth)
        total_obj = 0.0

        for i, original_h in enumerate(h_diags):
            # 1. Simulate using the NORMALIZED Hamiltonian
            probs = simulate_qaoa_fast(pars, train_n, norm_h_diags[i], depth)

            if is_supervised:
                # Maximize probability of known solutions
                total_obj -= np.sum(probs[sol_indices_list[i]])
            else:
                # 2. Calculate expected energy using the ORIGINAL Hamiltonian
                total_obj += np.sum(probs * original_h)

        return total_obj / len(h_diags)
    # 1. Grid Search
    dg_vals = np.linspace(-2.0, 2.0, 11)
    db_vals = np.linspace(0.1, 4.0, 11)

    best_obj = float('inf')
    best_deltas = [-0.8, 0.49]

    if not skip_grid:
        for dg in tqdm(dg_vals, desc="Grid Scanning"):
            for db in db_vals:
                obj_val = get_objective_value(dg, db)
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_deltas = [dg, db]

    print(f"  > Best Grid Start: dg={best_deltas[0]:.2f}, db={best_deltas[1]:.2f} (Obj={best_obj:.4f})")

    # 2. Refinement using COBYLA
    res = optimize.minimize(lambda d: get_objective_value(d[0], d[1]),
                            best_deltas, method='COBYLA', tol=1e-3, options={'maxiter': 200})

    # Format the output for readability
    final_val = -res.fun if is_supervised else res.fun
    print(f"  > LR QAOA Final {'Avg Prob' if is_supervised else 'Avg Energy'}: {final_val:.4f}")

    return get_lr_params(res.x, depth)
