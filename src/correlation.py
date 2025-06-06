import numpy as np
from scipy.spatial import cKDTree

def count_pairs(data1, data2, bins):
    tree1 = cKDTree(data1)
    tree2 = cKDTree(data2)
    return tree1.count_neighbors(tree2, bins)

def compute_two_point_correlation(galaxy_positions, bins, random_positions=None):
    """
    Compute the Landy-Szalay two-point correlation function.
    """
    if random_positions is None:
        x_min, y_min, z_min = np.min(galaxy_positions, axis=0)
        x_max, y_max, z_max = np.max(galaxy_positions, axis=0)
        num_random = 5 * len(galaxy_positions)

        random_positions = np.column_stack((
            np.random.uniform(x_min, x_max, num_random),
            np.random.uniform(y_min, y_max, num_random),
            np.random.uniform(z_min, z_max, num_random)
        ))

    DD = np.diff(count_pairs(galaxy_positions, galaxy_positions, bins))
    RR = np.diff(count_pairs(random_positions, random_positions, bins))
    DR = np.diff(count_pairs(galaxy_positions, random_positions, bins))

    n_g = len(galaxy_positions)
    n_r = len(random_positions)

    DD_norm = DD / (n_g * (n_g - 1) / 2)
    RR_norm = RR / (n_r * (n_r - 1) / 2)
    DR_norm = DR / (n_g * n_r)

    xi = (DD_norm - 2 * DR_norm + RR_norm) / RR_norm
    r = 0.5 * (bins[:-1] + bins[1:])

    return r, xi