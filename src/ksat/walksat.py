import numpy as np
from numba import njit
from tqdm import tqdm
import random


@njit
def fast_walksat_solver(n, clause_vars, clause_signs, max_flips, p_noise):
    """
    Numba-optimized WalkSAT solver.
    Operates on rigid 2D arrays instead of lists of tuples.
    """

    # Initialize assignment (0 or 1)
    assignment = np.random.randint(0, 2, n)

    # Pre-allocate array for unsatisfied clause indices to avoid list appending overhead
    m = clause_vars.shape[0]
    unsat_buffer = np.empty(m, dtype=np.int32)

    for flip in range(max_flips):

        # 1. Identify Unsatisfied Clauses (Fast Scan)
        unsat_count = 0
        for i in range(m):
            is_sat = False
            for k in range(8):  # K=8 (Fixed for 8-SAT)
                # var index
                v = clause_vars[i, k]
                # required sign (0 or 1)
                s = clause_signs[i, k]

                if assignment[v] == s:
                    is_sat = True
                    break

            if not is_sat:
                unsat_buffer[unsat_count] = i
                unsat_count += 1

        # 2. Check Success
        if unsat_count == 0:
            return flip + 1

        # 3. Pick Random Unsatisfied Clause
        # Numba doesn't support random.choice on arrays, so we use index
        rand_idx = np.random.randint(0, unsat_count)
        target_c_idx = unsat_buffer[rand_idx]

        # 4. Select Variable to Flip
        var_to_flip = -1

        # Random float check
        if np.random.random() < p_noise:
            # Random Walk: Pick random literal 0..7
            rand_lit_idx = np.random.randint(0, 8)
            var_to_flip = clause_vars[target_c_idx, rand_lit_idx]
        else:
            # Greedy: Minimize breaks
            # We must iterate over the 8 variables in the target clause
            best_var = -1
            min_breaks = 999999

            for k in range(8):
                candidate_var = clause_vars[target_c_idx, k]

                # Flip experimentally
                assignment[candidate_var] = 1 - assignment[candidate_var]

                # Count breaks (how many clauses become UNSAT)
                # Optimization: In heavy implementation, we'd use adjacency lists.
                # In this "Simple+Fast" version, we assume M is small enough for Numba to scan.
                current_breaks = 0

                # We assume we only care about the candidate's break score.
                # A full global scan is O(M), which is okay in Numba for M<2000.
                for i_scan in range(m):
                    # Check if clause i_scan is UNSAT
                    c_sat = False
                    for k_scan in range(8):
                        v_scan = clause_vars[i_scan, k_scan]
                        s_scan = clause_signs[i_scan, k_scan]
                        if assignment[v_scan] == s_scan:
                            c_sat = True
                            break
                    if not c_sat:
                        current_breaks += 1

                if current_breaks < min_breaks:
                    min_breaks = current_breaks
                    best_var = candidate_var

                # Flip back
                assignment[candidate_var] = 1 - assignment[candidate_var]

            var_to_flip = best_var

        # 5. Perform Flip
        assignment[var_to_flip] = 1 - assignment[var_to_flip]

    return max_flips


@njit
def walksatlm_paper_kernel(n, c_vars, c_signs, max_flips, p_noise, w1, w2):
    """
    Implements WalkSATlm exactly as described in Algorithm 1 of the paper.
    Uses 'num_true_lits' array to track clause states efficiently.
    """
    m = c_vars.shape[0]
    k_sat = c_vars.shape[1]  # 8

    # --- 1. Build Adjacency Structures ---
    # We need to look up which clauses a variable appears in,
    # AND what its sign is in that clause (to know if it's True or False).

    # Calculate degrees
    degrees = np.zeros(n, dtype=np.int32)
    for i in range(m):
        for k in range(k_sat):
            v = c_vars[i, k]
            degrees[v] += 1

    max_degree = 0
    for i in range(n):
        if degrees[i] > max_degree:
            max_degree = degrees[i]

    # Fill Adjacency: adj_vars[v] = [clause_idx, ...]
    #                 adj_signs[v] = [sign_in_clause, ...]
    adj_indices = np.full((n, max_degree), -1, dtype=np.int32)
    adj_signs = np.full((n, max_degree), -1, dtype=np.int32)
    current_fill = np.zeros(n, dtype=np.int32)

    for i in range(m):
        for k in range(k_sat):
            v = c_vars[i, k]
            s = c_signs[i, k]

            # Uniqueness check to handle variables appearing twice in a clause
            # (Though in standard benchmarks, usually variables are unique in clause)
            # We add it anyway for safety.
            pos = current_fill[v]
            adj_indices[v, pos] = i
            adj_signs[v, pos] = s
            current_fill[v] += 1

    # --- 2. Initialization ---
    assignment = np.random.randint(0, 2, n)

    # Track number of true literals per clause (Critical for Make_1 vs Make_2)
    num_true_lits = np.zeros(m, dtype=np.int32)

    for i in range(m):
        count = 0
        for k in range(k_sat):
            if assignment[c_vars[i, k]] == c_signs[i, k]:
                count += 1
        num_true_lits[i] = count

    # Buffer for unsatisfied clauses
    unsat_buffer = np.empty(m, dtype=np.int32)

    # --- 3. Main Loop ---
    for flip in range(1, max_flips + 1):

        # A. Find Unsatisfied Clauses
        unsat_count = 0
        for i in range(m):
            if num_true_lits[i] == 0:
                unsat_buffer[unsat_count] = i
                unsat_count += 1

        if unsat_count == 0:
            return flip  # Solved

        # B. Pick Clause C randomly
        rand_idx = np.random.randint(0, unsat_count)
        target_c_idx = unsat_buffer[rand_idx]

        # Candidates are variables in C
        candidates = c_vars[target_c_idx]

        # C. Calculate Properties for Candidates (Break, Make1, Make2)
        # We need to find "best" variable according to paper logic.

        # Paper logic:
        # 1. Identify set of vars with Break=0. If exists, pick max lmake.
        # 2. Else:
        #    - Prob p: Random
        #    - Prob 1-p: Min Break, tie-break max lmake.

        # To do this efficiently, we gather metrics for all candidates first.
        # Max K=8, so fixed size arrays are fast.

        cand_breaks = np.zeros(k_sat, dtype=np.int32)
        cand_lmakes = np.zeros(k_sat, dtype=np.int32)

        has_zero_break = False

        for k in range(k_sat):
            var = candidates[k]

            current_break = 0
            make_1 = 0
            make_2 = 0

            # Iterate clauses containing 'var'
            deg = current_fill[var]
            for idx in range(deg):
                c_idx = adj_indices[var, idx]
                s = adj_signs[var, idx]

                # Check clause state
                lit_count = num_true_lits[c_idx]

                # If var is currently True in this clause (assignment == s)
                # Flipping makes it False.
                if assignment[var] == s:
                    # If it was the ONLY true literal, we break the clause.
                    if lit_count == 1:
                        current_break += 1

                # If var is currently False in this clause (assignment != s)
                # Flipping makes it True.
                else:
                    # If clause was unsatisfied (0 true), it becomes 1-true. -> Make_1
                    if lit_count == 0:
                        make_1 += 1
                    # If clause was 1-true, it becomes 2-true. -> Make_2
                    elif lit_count == 1:
                        make_2 += 1

            cand_breaks[k] = current_break
            # lmake formula
            cand_lmakes[k] = (w1 * make_1) + (w2 * make_2)

            if current_break == 0:
                has_zero_break = True

        # D. Selection Logic
        best_var = -1

        if has_zero_break:
            # "If variable x in C with break(x)=0 then v <- x, breaking ties by greatest lmake"
            best_val = -1e9

            for k in range(k_sat):
                if cand_breaks[k] == 0:
                    score = cand_lmakes[k]
                    if score > best_val:
                        best_val = score
                        best_var = candidates[k]
                    elif score == best_val:
                        # Further ties random (Paper: "breaking ties by ... greatest lmake"
                        # does not specify 2nd tier ties, usually random)
                        if np.random.random() < 0.5:
                            best_var = candidates[k]

        else:
            # No zero break variable
            if np.random.random() < p_noise:
                # "With probability p: v <- a variable in C chosen at random"
                rand_k = np.random.randint(0, k_sat)
                best_var = candidates[rand_k]
            else:
                # "With probability 1-p: v <- variable with minimum break, breaking ties by greatest lmake"
                min_b = 999999
                max_l = -999999

                for k in range(k_sat):
                    b = cand_breaks[k]
                    l = cand_lmakes[k]

                    if b < min_b:
                        min_b = b
                        max_l = l
                        best_var = candidates[k]
                    elif b == min_b:
                        # Tie on break, check lmake
                        if l > max_l:
                            max_l = l
                            best_var = candidates[k]
                        elif l == max_l:
                            # Tie on both
                            if np.random.random() < 0.5:
                                best_var = candidates[k]

        # E. Perform Flip
        # We must update assignment AND num_true_lits for all affected clauses
        assignment[best_var] = 1 - assignment[best_var]

        deg = current_fill[best_var]
        for idx in range(deg):
            c_idx = adj_indices[best_var, idx]
            s = adj_signs[best_var, idx]

            # If sign matches new assignment, we became True (count +1)
            # If sign mismatches new assignment, we became False (count -1)
            if assignment[best_var] == s:
                num_true_lits[c_idx] += 1
            else:
                num_true_lits[c_idx] -= 1

    return max_flips
