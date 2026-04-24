import numpy as np
import random
from tqdm import tqdm


def run_walksat_specific(n, clauses, max_flips, p_noise):
    """
    WalkSAT solver adapted for the specific tuple structure: (vars_idx, signs).

    Args:
        n (int): Number of variables.
        clauses (list): List of tuples [(vars_idx, signs), ...].
        max_flips (int): Max steps.
        p_noise (float): Probability of random move.

    Returns:
        int: Number of flips used (or max_flips if failed).
    """

    # 1. Initialize Assignment (0 or 1)
    # Using 0-based indexing to match np.random.choice(n)
    assignment = np.random.randint(2, size=n)

    for flip in range(1, max_flips + 1):

        # --- A. Identify Unsatisfied Clauses ---
        unsat_indices = []

        for i, (vars_idx, signs) in enumerate(clauses):
            # A clause is SAT if at least one literal matches the assignment
            # signs[k] is the required value for assignment[vars_idx[k]]

            # Fast vectorized check:
            # vars_idx selects the current values from assignment
            # We compare them to signs. If any match, it's SAT.
            if not np.any(assignment[vars_idx] == signs):
                unsat_indices.append(i)

        # --- B. Check Success ---
        if not unsat_indices:
            return flip

        # --- C. Pick Random Unsatisfied Clause ---
        target_clause_idx = random.choice(unsat_indices)
        target_vars, target_signs = clauses[target_clause_idx]

        # --- D. Select Variable to Flip ---
        var_to_flip = -1

        if random.random() < p_noise:
            # Random Walk: Pick a random variable index from the clause
            var_to_flip = random.choice(target_vars)
        else:
            # Greedy Step: Find variable that minimizes total unsatisfied clauses
            best_var = target_vars[0]
            min_unsat = float('inf')

            # We only check variables present in the target clause
            # (Note: Using np.unique in case variables repeat in the clause)
            candidates = np.unique(target_vars)

            for var in candidates:
                # Flip experimentally
                assignment[var] = 1 - assignment[var]

                # Count current global unsatisfied (Linear scan - simple but slow)
                current_unsat_count = 0
                for v_idx, s in clauses:
                    if not np.any(assignment[v_idx] == s):
                        current_unsat_count += 1

                if current_unsat_count < min_unsat:
                    min_unsat = current_unsat_count
                    best_var = var

                # Flip back
                assignment[var] = 1 - assignment[var]

            var_to_flip = best_var

        # --- E. Perform Flip ---
        assignment[var_to_flip] = 1 - assignment[var_to_flip]

    return max_flips


def evaluate_model_walksat(dataset, max_flips=100000, p_noise=0.5):
    """
    Evaluates WalkSAT on the specific dataset format.
    """

    print(f"\n--- Comparative Evaluation: WalkSAT (p={p_noise}) ---")
    results = {'n': [], 'median_runtime': []}

    n_values = sorted(dataset.keys())

    for n in n_values:
        instances = dataset[n]
        runtimes = []

        for instance in tqdm(instances, desc=f"Benchmarking n={n}", leave=False):
            clauses = instance['clauses']

            # Pass N and the clauses list directly
            flips = run_walksat_specific(n, clauses, max_flips, p_noise)
            runtimes.append(flips)

        med_val = np.median(runtimes)
        results['n'].append(n)
        results['median_runtime'].append(med_val)

        tqdm.write(f"n={n}: Median Runtime = {med_val:.2f} flips")

    return results


def run_walksatlm_bench(n, clauses, max_flips, p_noise):
    """
    Executes WalkSATlm (Linear Make) on a single instance.

    Logic:
    1. If a variable has break=0 (Free Move), flip it.
    2. Else with prob p_noise, pick random variable from clause.
    3. Else (Greedy), pick variable with MINIMUM break.
       If ties exist, pick the one with MAXIMUM make (Linear Make).
    """

    # 0. Pre-processing: Build Adjacency List (Critical for speed)
    # adj[v] = list of clause indices where variable v appears
    adj = [[] for _ in range(n)]
    for i, (vars_idx, signs) in enumerate(clauses):
        unique_vars = np.unique(vars_idx)
        for v in unique_vars:
            adj[v].append(i)

    # 1. Initialization
    assignment = np.random.randint(2, size=n)

    # Track status of all clauses to avoid re-evaluating everything
    clause_status = np.zeros(len(clauses), dtype=bool)

    # Initial evaluation
    for i, (vars_idx, signs) in enumerate(clauses):
        if np.any(assignment[vars_idx] == signs):
            clause_status[i] = True

    # 2. Search Loop
    for flip in range(1, max_flips + 1):

        # A. Get Unsatisfied Clauses
        unsat_indices = np.where(~clause_status)[0]

        if len(unsat_indices) == 0:
            return flip  # Solved

        # B. Pick Random Unsatisfied Clause
        target_clause_idx = np.random.choice(unsat_indices)
        target_vars, _ = clauses[target_clause_idx]

        # Candidates are the variables in this clause
        candidates = np.unique(target_vars)

        # C. Evaluate Candidates (Compute Break & Make)
        # We need to find if there is a 'break=0' variable,
        # and otherwise gather metrics for tie-breaking.

        best_break_0_var = -1
        candidate_metrics = []  # List of (var, break, make)

        # Optimization: Only calculate metrics if we don't do a random walk immediately?
        # Standard WalkSATlm checks for break=0 *before* random walk.

        for var in candidates:
            current_break = 0
            current_make = 0

            # Flip experimentally
            assignment[var] = 1 - assignment[var]

            # Check only affected clauses
            for c_idx in adj[var]:
                c_vars, c_signs = clauses[c_idx]
                is_sat = np.any(assignment[c_vars] == c_signs)
                was_sat = clause_status[c_idx]

                if was_sat and not is_sat:
                    current_break += 1
                elif not was_sat and is_sat:
                    current_make += 1

            # Flip back
            assignment[var] = 1 - assignment[var]

            if current_break == 0:
                best_break_0_var = var
                break  # Found free move

            candidate_metrics.append((var, current_break, current_make))

        # D. Decision
        var_to_flip = -1

        if best_break_0_var != -1:
            var_to_flip = best_break_0_var
        elif random.random() < p_noise:
            # Random Walk
            var_to_flip = random.choice(candidates)
        else:
            # Greedy with Linear Make Tie-Breaking
            # Sort: Primary key = Break (Ascending), Secondary key = Make (Descending)
            candidate_metrics.sort(key=lambda x: (x[1], -x[2]))
            var_to_flip = candidate_metrics[0][0]

        # E. Perform Flip
        assignment[var_to_flip] = 1 - assignment[var_to_flip]

        # Update statuses efficiently
        for c_idx in adj[var_to_flip]:
            c_vars, c_signs = clauses[c_idx]
            clause_status[c_idx] = np.any(assignment[c_vars] == c_signs)

    return max_flips


def evaluate_model_walksatlm(dataset, max_flips=100000, p_noise=0.5):
    """
    Evaluates WalkSATlm (Linear Make) on the generated 8-SAT dataset.
    """

    print(f"\n--- Comparative Evaluation: WalkSATlm (p={p_noise}) ---")
    results = {'n': [], 'walksatlm_runtime': []}

    n_values = sorted(dataset.keys())

    for n in n_values:
        instances = dataset[n]
        runtimes = []

        for instance in tqdm(instances, desc=f"Benchmarking n={n}", leave=False):
            clauses = instance['clauses']

            # Run WalkSATlm
            flips = run_walksatlm_bench(n, clauses, max_flips, p_noise)
            runtimes.append(flips)

        med_runtime = np.median(runtimes)

        results['n'].append(n)
        results['walksatlm_runtime'].append(med_runtime)

        tqdm.write(f"n={n}: Median Runtime = {med_runtime:.2f} flips")

    return results
