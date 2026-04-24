import networkx as nx
import random
import numpy as np


def generate_dap_instance(num_nodes, edge_probability=0.5, weight_range=(1, 10), return_mwc=False):
    """
    Generates a random graph instance either for the Maximum Weighted Independent Set problem or the Maximum weighted
    clique problem.

    Args:
        num_nodes (int): The number of vertices in the graph.
        edge_probability (float): Probability for edge creation (0.0 to 1.0).
        weight_range (tuple): The (min, max) range for random integer node weights.
        return_mwc (bool): Whether to return the maximum weighted clique (MWC) instance or not.

    Returns:
        nx.Graph: A NetworkX graph with 'weight' attributes assigned to each node.
    """
    # Create a random Erdős-Rényi graph
    G = nx.erdos_renyi_graph(n=num_nodes, p=edge_probability)

    # Assign random weights to each node
    for node in G.nodes():
        G.nodes[node]['weight'] = random.randint(*weight_range)

    if return_mwc:
        G_complement = nx.complement(G)
        return G_complement

    return G


def graph_to_wcnf(G, output_file=None):
    """
    Translates an MWIS graph into a Max-2-SAT WCNF string.

    Args:
        G (nx.Graph): The graph with 'weight' attributes on nodes.
        output_file (str, optional): If provided, writes the WCNF to this file.

    Returns:
        str: The WCNF formatted string.
    """
    # DIMACS uses 1-based indexing for variables.
    # We create a mapping from NetworkX node IDs to 1-based integers.
    node_to_var = {node: i + 1 for i, node in enumerate(G.nodes())}

    num_vars = len(G.nodes())
    num_clauses = len(G.nodes()) + len(G.edges())

    # Calculate the "top weight" (infinity) for hard clauses.
    # It must be strictly greater than the sum of all soft clause weights.
    total_soft_weight = sum(data.get('weight', 1) for _, data in G.nodes(data=True))
    top_weight = total_soft_weight + 1

    lines = []
    # WCNF Header: p wcnf <num_variables> <num_clauses> <top_weight>
    lines.append(f"p wcnf {num_vars} {num_clauses} {top_weight}")

    # 1. Soft Clauses (Maximize node weights)
    # Format: <weight> <variable> 0
    for node, data in G.nodes(data=True):
        weight = data.get('weight', 1)
        var = node_to_var[node]
        lines.append(f"{weight} {var} 0")

    # 2. Hard Clauses (Adjacency constraints / Max-2-SAT)
    # Format: <top_weight> -<var1> -<var2> 0
    for u, v in G.edges():
        var_u = node_to_var[u]
        var_v = node_to_var[v]
        lines.append(f"{top_weight} -{var_u} -{var_v} 0")

    wcnf_string = "\n".join(lines)

    # Optional: Write to file for C++ solvers like UWrMaxSat
    if output_file:
        with open(output_file, 'w') as f:
            f.write(wcnf_string)

    return wcnf_string, node_to_var


def build_qubo_matrix(G, problem_type="MWIS", penalty=None):
    """
    Constructs the classical NxN QUBO matrix for MWIS or MWC.

    Args:
        G (nx.Graph): The graph instance (nodes must have 'weight' attributes).
        problem_type (str): Either "MWIS" or "MWC".
        penalty (float): Penalty multiplier for breaking constraints.

    Returns:
        np.ndarray: An NxN upper-triangular matrix representing the QUBO.
    """
    num_nodes = G.number_of_nodes()
    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    if penalty is None:
        penalty = sum(nx.get_node_attributes(G, 'weight').values()) + 1.0

    # Initialize the NxN classical matrix
    Q = np.zeros((num_nodes, num_nodes))
    weights = nx.get_node_attributes(G, 'weight')

    # 1. Fill the Diagonal (Objective / Weights)
    # Since x_i^2 = x_i for binary variables, linear terms go on the diagonal
    for node, w_i in weights.items():
        i = node_to_idx[node]
        # We want to maximize weight, so we minimize negative weight
        Q[i, i] = -w_i

    # 2. Fill the Off-Diagonal (Constraints / Penalties)
    if problem_type.upper() == "MWIS":
        penalty_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    elif problem_type.upper() == "MWC":
        G_comp = nx.complement(G)
        penalty_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G_comp.edges()]

    for i, j in penalty_edges:
        u, v = min(i, j), max(i, j)  # Ensure upper-triangular format
        # Place the constraint penalty in the off-diagonal
        Q[u, v] = penalty

    return Q


def qubo_to_ising(Q, return_matrix=False, return_2d=False):
    """
    Converts a classical upper-triangular QUBO matrix into a Quantum Ising Hamiltonian.

    Args:
        Q (np.ndarray): An NxN upper-triangular classical QUBO matrix.
        return_matrix (bool): If True, returns the Hamiltonian as a NumPy array.
                              If False, returns a dictionary of Pauli coefficients.
        return_2d (bool): If True (and return_matrix is True), returns a dense 2^N x 2^N matrix.
                          If False, returns a 1D array of length 2^N (highly recommended for memory).

    Returns:
        dict or np.ndarray: The Hamiltonian representation.
    """
    num_nodes = Q.shape[0]

    # --- MATRIX OUTPUT PATH ---
    if return_matrix:
        dim = 2 ** num_nodes
        diagonal = np.zeros(dim)

        # The diagonal of the quantum Hamiltonian is exactly the classical
        # QUBO energy evaluated for every possible bitstring state!
        for state in range(dim):
            # Convert the integer state into a binary array 'x'
            # (e.g., state 5 for N=3 becomes x = [1, 0, 1])
            x = np.array([(state >> i) & 1 for i in range(num_nodes)])

            # The energy is mathematically just x^T * Q * x
            diagonal[state] = x.T @ Q @ x

        if return_2d:
            # Warning: Scales poorly with memory for N > 12
            return np.diag(diagonal)

        return diagonal

    # --- DICTIONARY OUTPUT PATH ---
    linear_terms = {i: 0.0 for i in range(num_nodes)}
    quadratic_terms = {}
    offset = 0.0

    for i in range(num_nodes):
        # 1. Process the Diagonal (Classical linear terms Q_ii)
        offset += Q[i, i] / 2.0
        linear_terms[i] -= Q[i, i] / 2.0

        # 2. Process the Off-Diagonal (Classical quadratic constraints Q_ij)
        for j in range(i + 1, num_nodes):
            if Q[i, j] != 0.0:
                offset += Q[i, j] / 4.0

                # The off-diagonal term affects both connected qubits linear terms
                linear_terms[i] -= Q[i, j] / 4.0
                linear_terms[j] -= Q[i, j] / 4.0

                # Create the quantum quadratic term Z_i * Z_j
                quadratic_terms[(i, j)] = Q[i, j] / 4.0

    # Clean up linear terms that sum exactly to zero
    linear_terms = {k: v for k, v in linear_terms.items() if abs(v) > 1e-8}

    return {
        "linear": linear_terms,
        "quadratic": quadratic_terms,
        "offset": offset
    }


if __name__ == "__main__":
    mwis = generate_dap_instance(num_nodes=10, edge_probability=0.5, return_mwc=False)
    mwis_Q = build_qubo_matrix(mwis)

    max_2_sat = graph_to_wcnf(mwis)

    # 1. Get the terms for building a quantum circuit
    hamiltonian_dict = qubo_to_ising(mwis_Q)

    # 2. Get the 1D diagonal array for efficient statevector simulation
    hamiltonian_1d = qubo_to_ising(mwis_Q, return_matrix=True)

    # 3. Get the full 2D matrix if a specific math library demands it
    hamiltonian_2d = qubo_to_ising(mwis_Q, return_matrix=True, return_2d=True)

    print(hamiltonian_1d)
    print(hamiltonian_2d)
    print(hamiltonian_dict)
    print(mwis)
    print(max_2_sat)
