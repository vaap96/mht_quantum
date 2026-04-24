import numpy as np


def generate_8sat_clauses(n, m, k_sat=8):
    """Generates random 8-SAT clauses.
        output list of tupples (indexes,sign)"""

    clauses = []
    for _ in range(m):
        vars_idx = np.random.choice(n, k_sat, replace=True)
        literals = np.random.randint(0, 2, size=k_sat)
        clauses.append((vars_idx, literals))

    return clauses


def get_exact_sat_solutions(n, clauses):
    """Finds exact solutions (Energy = 0) classically.
        output bitstring of sols"""

    num_states = 2**n
    violation_counts = np.zeros(num_states, dtype=np.int32)
    states = np.arange(num_states, dtype=np.int32)

    for vars_idx, is_negated_list in clauses:
        clause_violated_mask = np.ones(num_states, dtype=bool)
        for i, v_idx in enumerate(vars_idx):
            target_bit = int(is_negated_list[i])
            bit_val = (states >> v_idx) & 1
            clause_violated_mask &= (bit_val == target_bit)
        violation_counts[clause_violated_mask] += 1

    return np.where(violation_counts == 0)[0]


def get_hamiltonian_diagonal(n, clauses):
    """Precomputes the diagonal of H_C."""
    num_states = 2**n
    diagonal = np.zeros(num_states, dtype=np.float64)
    states = np.arange(num_states, dtype=np.int32)
    for vars_idx, literals in clauses:
        violation_mask = np.ones(num_states, dtype=bool)
        for i, v_idx in enumerate(vars_idx):
            target_bit = literals[i]
            bit_val = (states >> v_idx) & 1
            violation_mask &= (bit_val == target_bit)
        diagonal[violation_mask] += 1.0

    return diagonal


def generate_training_set(train_n, train_size, clause_density, k_sat=8):
    print(f"Generating training set (n={train_n}, size={train_size})...")
    data = []
    lambda_val = clause_density * train_n
    while len(data) < train_size:
        m = max(1, np.random.poisson(lam=lambda_val))
        clauses = generate_8sat_clauses(train_n, m, k_sat)
        sol_indices = get_exact_sat_solutions(train_n, clauses)
        if len(sol_indices) > 0:
            h_diag = get_hamiltonian_diagonal(train_n, clauses)
            data.append((h_diag, sol_indices))

    return data


def generate_benchmark_instances(n_range, test_size, clause_density, k_sat=8):
    dataset = {}

    print(f"\n--- Generating Benchmark Dataset (n={list(n_range)}) ---")

    for n in n_range:
        lambda_val = clause_density * n
        instances = []
        valid_count = 0

        with tqdm(total=test_size, desc=f"Generating n={n}", leave=False) as pbar:
            while valid_count < test_size:
                # Generate random clause count based on Poisson distribution
                m = max(1, np.random.poisson(lam=lambda_val))
                clauses = generate_8sat_clauses(n, m, k_sat)

                # Identify solutions; filtering for SAT instances only
                sol_indices = get_exact_sat_solutions(n, clauses)

                if len(sol_indices) == 0:
                    continue

                # Pre-compute Hamiltonian diagonal to save time during evaluation
                h_diag = get_hamiltonian_diagonal(n, clauses)

                instances.append({
                    'clauses': clauses,
                    'sol_indices': sol_indices,
                    'h_diag': h_diag
                })

                valid_count += 1
                pbar.update(1)

        dataset[n] = instances

    return dataset
