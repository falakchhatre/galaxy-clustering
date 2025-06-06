import numpy as np
from scipy.spatial import cKDTree

def count_pairs(data1, data2, bins):
    tree1 = cKDTree(data1)
    tree2 = cKDTree(data2)  # build once, reuse!

    counts = np.zeros(len(bins) - 1, dtype=int)
    for i in range(len(bins) - 1):
        counts[i] = tree1.count_neighbors(tree2, bins[i+1]) - tree1.count_neighbors(tree2, bins[i])

    return counts

def compute_two_point_correlation(galaxy_positions, bins):
    """
    Compute the Landy-Szalay two-point correlation function.
    """
    # Define random sample within galaxy bounding box
    x_min, y_min, z_min = np.min(galaxy_positions, axis=0)
    x_max, y_max, z_max = np.max(galaxy_positions, axis=0)
    num_random = 5 * len(galaxy_positions)
    
    random_positions = np.column_stack((
        np.random.uniform(x_min, x_max, num_random),
        np.random.uniform(y_min, y_max, num_random),
        np.random.uniform(z_min, z_max, num_random)
    ))
    
    DD = count_pairs(galaxy_positions, galaxy_positions, bins)
    RR = count_pairs(random_positions, random_positions, bins)
    DR = count_pairs(galaxy_positions, random_positions, bins)

    DD_norm = DD / (len(galaxy_positions) * (len(galaxy_positions) - 1) / 2)
    RR_norm = RR / (num_random * (num_random - 1) / 2)
    DR_norm = DR / (len(galaxy_positions) * num_random)

    xi = (DD_norm - 2 * DR_norm + RR_norm) / RR_norm
    r = 0.5 * (bins[:-1] + bins[1:])
    
    return r, xi
