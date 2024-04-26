import numpy as np
import networkx as nx
import csv
from tqdm import tqdm
from scipy.stats import spearmanr


def load_network(path):
    """Loads a network dataset from a CSV file and constructs its supra adjacency matrices.

    Args:
        path: str
            The file path to the CSV dataset. For example, "folder/data.csv".

    Returns:
        network: numpy.ndarray
            A (g * (n * n)) dimensional ndarray representing g layers of supra adjacency matrices.
        shape: tuple
            A tuple with two elements that describes the structure of the network, where:
                - g (shape[0]) represents the number of layers in the network.
                - n (shape[1]) represents the number of nodes in each layer.
    """

    filename = open(path, 'r')
    file = csv.reader(filename)

    layers = set()
    nodes = set()
    for row in file:
        layers.add(int(row[0]))
        nodes.add(int(row[1]))
        nodes.add(int(row[2]))

    g = max(layers)
    n = max(nodes)
    shape = (g, n)

    network = np.zeros((g, g), dtype=object)

    for row in range(g):
        for col in range(g):
            network[row][col] = np.zeros((n, n), dtype=float)

    filename = open(path, 'r')
    file = csv.reader(filename)

    for row in file:
        currLayer = int(row[0]) - 1
        network[currLayer][currLayer][int(row[1]) - 1][int(row[2]) - 1] = float(row[3])
        network[currLayer][currLayer][int(row[2]) - 1][int(row[1]) - 1] = float(row[3])

    return network, shape


def layer_plex_rank(network, shape, a, s, gamma, delta=0.85, error = 1e-6):
    """Calculates node centralities and layer influence in a multiplex network.

    Args:
        network: numpy.ndarray
            A (g * (n * n)) dimensional ndarray representing g layers of supra adjacency matrices.
        shape: tuple
            A tuple with two elements indicating the structure of the network, where:
                - g (shape[0]) represents the number of layers.
                - n (shape[1]) represents the number of nodes per layer.
        a: int
            Determines the influence of a layer based on its weight (W[layer]). Values:
                - 1: Influence is proportional to W[layer].
                - 0: Influence is normalized with respect to W[layer].
        s: int
            Modifies layer influence based on the centrality of nodes within the layer. Values:
                - 1: Layers with more central nodes have greater influence.
                - -1: Layers with fewer highly central nodes have greater influence.
        gamma: float
            Adjusts the contribution of nodes based on their centrality for the calculation of Z. Conditions: gamma > 0
                - gamma > 1: Enhances the contribution of nodes with low centrality.
                - gamma < 1: Suppresses the contribution of nodes with low centrality.
        delta: float
            The damping factor, typically set to 0.85, used within the context of PageRank algorithms.
        error: float
            The error tolerance for stopping iterations, initially set to 1e-6 (0.000001).
    """

    g = shape[0]
    n = shape[1]
    A = np.diagonal(network)

    W = np.zeros(g, dtype=float)
    for layer in range(g):
        W[layer] = np.sum(A[layer])

    B = np.zeros((g, n), dtype=float)
    for layer in range(g):
        B[layer] = np.sum(A[layer], axis=0) / (W[layer] + 1)

    # Centrality of layer initialized as 0.
    # Z = np.random.uniform(0, 1, g)
    # Set the initial Z value as a constant to avoid random effects when calculation
    # from (gamma = 0.1) to (gamma = 3.0).
    Z = np.full(g, 0.5)

    G = np.zeros((n, n), dtype=float)
    for layer in range(g):
        G += A[layer] * Z[layer]

    # Centrality of node initialized as 0.
    # X = np.random.uniform(0, 1, n)
    # Set the initial X value as a constant to avoid random effects when calculation
    # from (gamma = 0.1) to (gamma = 3.0).
    X = np.full(n, 0.5)

    # V_i initialized as 0.
    V = np.zeros(n, dtype=float)
    # V_i = \sum^{g}_{j=1} [G_{ij} + G_{ji}
    V = np.sum(G, axis=1) + np.sum(G, axis=0)
    # Apply theta (Heaviside step function) to array V.
    V = np.where(V <= 0, 0, V)

    # Iterations stop when related error is less than setting error.
    while True:
        beta = np.sum((1 - delta * (np.sum(G, axis=0) > 1)) * X) / np.sum(V)

        X_constant = X
        X = np.zeros(n, dtype=float)
        for layer in range(g):
            X += B[layer] * X_constant / (np.sum(B[layer]) + 1)
        X += V * beta
        X /= np.sum(X)

        Z = np.zeros(g, dtype=float)
        # Handle with ZeroDivisionError for (0 ** -1).
        X_not_zero = np.copy(X)
        X_not_zero[X_not_zero == 0] = 1
        for layer in range(g):
            Z[layer] = (W[layer] ** a) * (np.sum(B[layer] * (X_not_zero ** (s * gamma))) ** s)
        Z /= np.sum(Z)

        G = np.zeros((n, n), dtype=float)
        for layer in range(g):
            G += A[layer] * Z[layer]

        V = np.zeros(n, dtype=float)
        V = np.sum(G, axis=1) + np.sum(G, axis=0)
        V = np.where(V <= 0, 0, V)

        # Stopping condition.
        if np.average(np.absolute(X - X_constant)) < error:
            break

    return X, Z


def benchmark_centrality(network, shape, method):
    """Calculates the centrality of each node in a network using NetworkX.

    Args:
        network: numpy.ndarray
            A (g * (n * n)) dimensional ndarray representing g layers of supra adjacency matrices.
        shape: tuple
            A tuple with two elements, where:
                - g (shape[0]) represents the number of layers in the network.
                - n (shape[1]) represents the number of nodes per layer.
        method: str
            Specifies the centrality calculation method to be used. Valid options are:
                - "betweenness": Betweenness centrality
                - "closeness": Closeness centrality
                - "degree": Degree centrality
                - "eigenvector": Eigenvector centrality
                - "pagerank": PageRank centrality

    Returns:
        nodes_centralities: numpy.ndarray
            An array containing the centrality values of the nodes.
        layers_influences: numpy.ndarray
            An array containing the influence values of the layers.
    """

    layers_influences = np.zeros(shape[0], dtype=float)
    nodes_centralities = np.zeros(shape[1], dtype=float)
    for layer in range(shape[0]):
        G = nx.from_numpy_array(network[layer][layer])
        match method:
            case 'betweenness':
                T = nx.betweenness_centrality(G, normalized=True)
            case 'closeness':
                T = nx.closeness_centrality(G)
            case 'degree':
                T = nx.degree_centrality(G)
            case 'eigenvector':
                T = nx.eigenvector_centrality_numpy(G)
            case 'pagerank':
                T = nx.pagerank(G)
            case _:
                raise TypeError("Invalid benchmark method name.")
        layers_influences[layer] = sum(list(T.values()))
        for node in range(shape[1]):
            nodes_centralities[node] += T[node]
    return nodes_centralities, layers_influences


def generate_rank_by_centrality(centrality_values):
    """Generates a ranking of nodes based on their centrality values.

    Args:
        X: numpy.ndarray
            An array of centrality values for each node.

    Returns:
        ranking: numpy.ndarray
            An array of rankings for each node based on their centrality values.
    """
    indices = np.argsort(-centrality_values)
    ranking = np.empty_like(indices)
    ranking[indices] = np.arange(len(-centrality_values))
    return ranking


def calculate_spearman_for_lists(lists_dict, show=False):
    """Calculate and save Spearman rank-order correlation coefficients and p-values for every pair of lists provided in a dictionary.

    Args:
        lists_dict: dict
            A dictionary where each key is a string representing the name of a list, and its corresponding value is the list itself.

    Returns:
        tuple: Two matrices, one for the correlation coefficients and the other for the p-values.

    Example usage:
        lists_dict = {
            'X': [1, 2, 3, 4, 5, 6],
            'bX': [5, 6, 7, 8, 9, 10],
            'cX': [2, 3, 4, 5, 6, 7],
            'dX': [5, 4, 3, 2, 1, 0],
            'eX': [9, 8, 7, 6, 5, 4],
            'pX': [1, 2, 3, 4, 5, 6]
        }
        correlations, p_values = calculate_spearman_for_lists(lists_dict)
    """

    names = list(lists_dict.keys())
    lists = list(lists_dict.values())
    n = len(lists)

    correlation_matrix = np.zeros((n, n))
    p_value_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):  # Start from i to avoid recalculating duplicates
            if i == j:
                correlation = 1.0
                p_value = 0.0
            else:
                correlation, p_value = spearmanr(lists[i], lists[j])

            correlation_matrix[i, j] = correlation_matrix[j, i] = correlation
            p_value_matrix[i, j] = p_value_matrix[j, i] = p_value
            if show:
                print(f"Spearman correlation between {names[i]} and {names[j]}: {correlation:.5f} (p-value: {p_value:.3f})")
    return correlation_matrix, p_value_matrix


def loocv(files_list, origin_network, origin_shape, method, a=1, s=1, gamma=1, sort=False):
    """Performs Leave-One-Out Cross-Validation (LOOCV) on a list of subnetworks to assess the stability of centrality measures.

    Args:
        files_list: list
            A list of paths to network dataset files.
        origin_network: numpy.ndarray
            The original network represented as a numpy array.
        origin_shape: tuple
            A tuple indicating the shape of the original network array, where:
                - origin_shape[0] represents the number of layers in the network.
                - origin_shape[1] represents the number of nodes per layer.
        method: str
            The centrality measure to be applied. Valid options are:
                - "LayerPlexRank": Custom centrality measure incorporating parameters `a`, `s`, `gamma`.
                - "betweenness": Betweenness centrality.
                - "closeness": Closeness centrality.
                - "degree": Degree centrality.
                - "eigenvector": Eigenvector centrality.
        a: float, optional
            Parameter 'a' specific to LayerPlexRank, default is 1.
        s: float, optional
            Parameter 's' specific to LayerPlexRank, default is 1.
        gamma: float, optional
            Parameter 'gamma' specific to LayerPlexRank, default is 1.
        sorted: bool, optional
            If True, returns the percentage differences sorted in ascending order. Default is False.

    Returns:
        percentage_diff: numpy.ndarray:
            An array of percentage differences between the original centrality values and the average centrality values from LOOCV.

    Raises:
        TypeError:
            If an invalid method is specified.
    """

    avg_rank_X = np.zeros(origin_shape[1], dtype=float)

    for path_sub_dataset in tqdm(files_list):
        sub_network, sub_shape = load_network(path_sub_dataset)
        match method:
            case 'LayerPlexRank':
                X, Z = layer_plex_rank(origin_network, origin_shape, a, s, gamma)
                sub_X, sub_Z = layer_plex_rank(sub_network, sub_shape, a, s, gamma)
            case 'betweenness':
                X, Z = benchmark_centrality(origin_network, origin_shape, 'betweenness')
                sub_X, sub_Z = benchmark_centrality(sub_network, sub_shape, 'betweenness')
            case 'closeness':
                X, Z = benchmark_centrality(origin_network, origin_shape, 'closeness')
                sub_X, sub_Z = benchmark_centrality(sub_network, sub_shape, 'closeness')
            case 'degree':
                X, Z = benchmark_centrality(origin_network, origin_shape, 'degree')
                sub_X, sub_Z = benchmark_centrality(sub_network, sub_shape, 'degree')
            case 'eigenvector':
                X, Z = benchmark_centrality(origin_network, origin_shape, 'eigenvector')
                sub_X, sub_Z = benchmark_centrality(sub_network, sub_shape, 'eigenvector')
            case 'pagerank':
                X, Z = benchmark_centrality(origin_network, origin_shape, 'pagerank')
                sub_X, sub_Z = benchmark_centrality(sub_network, sub_shape, 'pagerank')
            case _:
                raise TypeError("Invalid benchmark method name.")
        while len(sub_X) != origin_shape[1]:
            sub_X = np.append(sub_X, 0.0)
        sorted_indices = np.argsort(-sub_X)
        V_sub_X = np.empty_like(sorted_indices)
        V_sub_X[sorted_indices] = np.arange(origin_shape[1])
        avg_rank_X += V_sub_X
    avg_rank_X /= origin_shape[1]

    V_X = generate_rank_by_centrality(X)

    difference_X = V_X - avg_rank_X
    percentage_diff_X = abs(difference_X) / len(difference_X)

    if sort:
        return sorted(percentage_diff_X)
    else:
        return percentage_diff_X


def parameter_sensitivity(network, shape, node_ids):
    results = []
    s_values = [1]
    a_values = [1]
    gamma_values = np.arange(0, 3.1, 0.1)

    for s in s_values:
        for a in a_values:
            for gamma in tqdm(gamma_values):
                X, Z = layer_plex_rank(network, shape, a, s, gamma)
                rank = generate_rank_by_centrality(X)
                for node_id in node_ids:
                    results.append({
                        's': s,
                        'a': a,
                        'gamma': gamma,
                        'node_id': node_id,
                        'centrality': X[node_id - 1],
                        'rank': rank[node_id - 1] + 1
                    })

    return results