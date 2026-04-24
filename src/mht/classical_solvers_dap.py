import time
import numpy as np
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2


def solve_wcnf_instance(wcnf_path_or_string, is_file=True):
    """
    Loads and solves a WCNF Max-2-SAT instance.
    It's an exact solver.
    Returns:
        selected_vars: true variables (solution)
        max_weight: weight kept
    """
    # Load the WCNF formula
    if is_file:
        print(f"Loading WCNF from file: {wcnf_path_or_string}")
        formula = WCNF(from_file=wcnf_path_or_string)
    else:
        print("Loading WCNF from string...")
        formula = WCNF(from_string=wcnf_path_or_string)

    # Initialize and run the RC2 solver
    print("Solving with RC2...")
    with RC2(formula) as solver:
        start_time = time.perf_counter()

        model = solver.compute()

        end_time = time.perf_counter()

        execution_time = end_time - start_time

        if model is None:
            print("No solution found. The hard clauses (edges) are unsatisfiable.")
            print(f"Time to Solution: {execution_time:.6f} seconds")
            return None, None

        # Parse the Model (find True variables)
        selected_vars = [var for var in model if var > 0]

        # Calculate the Objective Value (weight of solvers discarded)
        penalty_cost = solver.cost

        # subtract the penalty from the total soft weights to find max weight kept
        total_possible_weight = sum(formula.wght)
        max_weight = total_possible_weight - penalty_cost

        print("\n--- Results ---")
        print(f"Selected Variables: {selected_vars}")
        print(f"Penalty Cost:       {penalty_cost}")
        print(f"Maximized Weight:   {max_weight}")
        print(f"Time to Solution: {execution_time:.6f} seconds")

        return selected_vars, max_weight


def max2sat_walksat_kernel(n_total, c_vars, c_signs, c_weights, max_flips, p_noise, dummy_var):
    """
    Highly optimized NumPy WalkSAT kernel for Max-2-SAT.
    """
    m = c_vars.shape[0]
    k_sat = 2

    # --- 1. Build Adjacency Structures ---
    degrees = np.zeros(n_total, dtype=np.int32)
    for i in range(m):
        for k in range(k_sat):
            v = c_vars[i, k]
            degrees[v] += 1

    max_degree = np.max(degrees) if n_total > 0 else 0

    adj_indices = np.full((n_total, max_degree), -1, dtype=np.int32)
    adj_signs = np.full((n_total, max_degree), -1, dtype=np.int32)
    current_fill = np.zeros(n_total, dtype=np.int32)

    for i in range(m):
        for k in range(k_sat):
            v = c_vars[i, k]
            s = c_signs[i, k]
            pos = current_fill[v]
            adj_indices[v, pos] = i
            adj_signs[v, pos] = s
            current_fill[v] += 1

    # --- 2. Zero-State Initialization ---
    # Start with an empty set (0 broken hard clauses)
    assignment = np.zeros(n_total, dtype=np.int32)
    num_true_lits = np.zeros(m, dtype=np.int32)

    for i in range(m):
        count = 0
        for k in range(k_sat):
            if assignment[c_vars[i, k]] == c_signs[i, k]:
                count += 1
        num_true_lits[i] = count

    current_cost = np.sum(c_weights[num_true_lits == 0])
    best_cost = current_cost
    best_assignment = assignment.copy()

    # --- 3. Main Loop ---
    for flip in range(1, max_flips + 1):
        unsat_mask = (num_true_lits == 0)
        unsat_count = np.sum(unsat_mask)

        if unsat_count == 0:
            best_cost = 0
            best_assignment = assignment.copy()
            break

        unsat_indices = np.nonzero(unsat_mask)[0]

        # Pick a random unsatisfied clause
        target_c_idx = np.random.choice(unsat_indices)

        # Get variables, but completely IGNORE the dummy phantom variable
        candidates = np.unique(c_vars[target_c_idx])
        candidates = candidates[candidates != dummy_var]

        # Selection Logic
        best_var = -1

        if np.random.random() < p_noise:
            best_var = np.random.choice(candidates)
        else:
            best_delta = float('inf')

            for var in candidates:
                delta_cost = 0
                deg = current_fill[var]

                for idx in range(deg):
                    c_idx = adj_indices[var, idx]
                    s = adj_signs[var, idx]
                    weight = c_weights[c_idx]

                    if assignment[var] == s:
                        if num_true_lits[c_idx] == 1:
                            delta_cost += weight
                    else:
                        if num_true_lits[c_idx] == 0:
                            delta_cost -= weight

                if delta_cost < best_delta:
                    best_delta = delta_cost
                    best_var = var
                elif delta_cost == best_delta:
                    if np.random.random() < 0.5:
                        best_var = var

        # Perform Flip
        assignment[best_var] = 1 - assignment[best_var]

        # Update State
        deg = current_fill[best_var]
        for idx in range(deg):
            c_idx = adj_indices[best_var, idx]
            s = adj_signs[best_var, idx]
            weight = c_weights[c_idx]

            if assignment[best_var] == s:
                if num_true_lits[c_idx] == 0:
                    current_cost -= weight
                num_true_lits[c_idx] += 1
            else:
                num_true_lits[c_idx] -= 1
                if num_true_lits[c_idx] == 0:
                    current_cost += weight

                    # Track Global Best
        if current_cost < best_cost:
            best_cost = current_cost
            best_assignment[:] = assignment[:]

    return best_assignment, best_cost


def wcnf_walksat(wcnf_string, max_flips=10000, p_noise=0.2):
    """
    Parses a WCNF string and runs the fast WalkSAT kernel using a phantom dummy variable.
    """
    lines = [line.strip() for line in wcnf_string.strip().split('\n') if line.strip() and not line.startswith('c')]

    header = lines[0].split()
    n = int(header[2])
    top_weight = int(header[4])
    m = len(lines) - 1

    # We allocate 1 extra variable as a phantom padder (Index n, since it's 0-indexed)
    dummy_var = n
    n_total = n + 1

    c_vars = np.zeros((m, 2), dtype=np.int32)
    c_signs = np.zeros((m, 2), dtype=np.int32)
    c_weights = np.zeros(m, dtype=np.int32)
    total_soft_weight = 0

    for i, line in enumerate(lines[1:]):
        parts = list(map(int, line.split()))
        weight = parts[0]
        literals = parts[1:-1]

        c_weights[i] = weight
        if weight < top_weight:
            total_soft_weight += weight

        if len(literals) == 1:
            # Soft clause: Pad with the dummy variable.
            # We set the dummy sign to 1, but its value is locked at 0. It will never satisfy.
            var = abs(literals[0]) - 1
            sign = 1 if literals[0] > 0 else 0

            c_vars[i, 0] = var;
            c_signs[i, 0] = sign
            c_vars[i, 1] = dummy_var;
            c_signs[i, 1] = 1

        elif len(literals) == 2:
            # Hard clause
            var0 = abs(literals[0]) - 1
            sign0 = 1 if literals[0] > 0 else 0
            var1 = abs(literals[1]) - 1
            sign1 = 1 if literals[1] > 0 else 0

            c_vars[i, 0] = var0;
            c_signs[i, 0] = sign0
            c_vars[i, 1] = var1;
            c_signs[i, 1] = sign1

    # --- Run Timed Kernel ---
    start_time = time.perf_counter()
    best_assignment, best_cost = max2sat_walksat_kernel(
        n_total, c_vars, c_signs, c_weights, max_flips, p_noise, dummy_var
    )
    exec_time = time.perf_counter() - start_time

    # --- Format Output ---
    # Strip out the dummy variable and map back to 1-indexed variables
    selected_vars = [i + 1 for i, val in enumerate(best_assignment) if val == 1 and i != dummy_var]
    maximized_weight = total_soft_weight - best_cost

    return selected_vars, maximized_weight, best_cost, exec_time
