"""
final_submission.py

A python script for galaxy clustering and redshift-space distortion analysis.
Includes:
- Data loading and coordinate conversion
- Redshift space distortion modeling
- Two-point correlation function calculation
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from astropy.coordinates import SkyCoord, spherical_to_cartesian
from astropy.cosmology import Planck18
import astropy.units as u
from typing import Optional, Tuple


#data_loader.py
def load_raw_galaxy_catalog(path: str) -> pd.DataFrame:
    """
    Load raw galaxy data from CSV and drop unnecessary columns.
    """
    df = pd.read_csv(path)
    if 'zWarning' in df.columns:
        df = df.drop(columns=['zWarning'])
    return df

def convert_to_cartesian_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert RA, Dec, and redshift into Cartesian coordinates.
    """
    coords = SkyCoord(ra=df['ra'].values * u.degree,
                      dec=df['dec'].values * u.degree,
                      frame='icrs')
    comoving_distances = Planck18.comoving_distance(df['z'].values)
    
    ra_rad = coords.ra.radian
    dec_rad = coords.dec.radian
    r = comoving_distances.value 
    
    df['x'] = r * np.cos(dec_rad) * np.cos(ra_rad)
    df['y'] = r * np.cos(dec_rad) * np.sin(ra_rad)
    df['z_cart'] = r * np.sin(dec_rad)
    return df

def load_cartesian_galaxy_data(path: str) -> pd.DataFrame:
    """
    Load precomputed Cartesian galaxy data from CSV.
    """
    return pd.read_csv(path)

#rsd.py
def apply_redshift_space_distortions(df: pd.DataFrame, velocity_dispersion_kms: float = 300.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply line-of-sight velocity distortions to redshifts and compute distorted Cartesian coordinates.
    """
    c_kms = 3e5  # speed of light in km/s
    
    peculiar_velocities = np.random.normal(0, velocity_dispersion_kms, len(df))
    
    delta_z = (peculiar_velocities / c_kms) * (1 + df['z'].values)
    z_distorted = df['z'].values + delta_z
    
    if np.any(z_distorted < 0):
        raise ValueError("Negative redshift after velocity distortion.")
    
    distorted_distances = Planck18.comoving_distance(z_distorted).to(u.Mpc).value
    
    ra_rad = np.deg2rad(df['ra'].values)
    dec_rad = np.deg2rad(df['dec'].values)
    
    x, y, z = spherical_to_cartesian(distorted_distances, dec_rad, ra_rad)
    
    return x.value, y.value, z.value

#correlation.py
def count_pairs_within_bins(data_points_1: np.ndarray, data_points_2: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Count pairs within distance bins between two point sets using KD-Trees.
    """
    tree1 = cKDTree(data_points_1)
    tree2 = cKDTree(data_points_2)
    counts = tree1.count_neighbors(tree2, bin_edges)
    return counts

def compute_two_point_correlation_function(galaxy_positions: np.ndarray, bin_edges: np.ndarray, random_positions: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Landy-Szalay two-point correlation function.
    """
    if random_positions is None:
        x_min, y_min, z_min = np.min(galaxy_positions, axis=0)
        x_max, y_max, z_max = np.max(galaxy_positions, axis=0)
        n_random = 5 * len(galaxy_positions)
        random_positions = np.column_stack((
            np.random.uniform(x_min, x_max, n_random),
            np.random.uniform(y_min, y_max, n_random),
            np.random.uniform(z_min, z_max, n_random)
        ))
    else:
        n_random = len(random_positions)
    
    n_galaxies = len(galaxy_positions)
    
    # Count pairs
    DD_counts = np.diff(count_pairs_within_bins(galaxy_positions, galaxy_positions, bin_edges))
    RR_counts = np.diff(count_pairs_within_bins(random_positions, random_positions, bin_edges))
    DR_counts = np.diff(count_pairs_within_bins(galaxy_positions, random_positions, bin_edges))
    
    # Normalize counts
    DD_norm = DD_counts / (n_galaxies * (n_galaxies - 1) / 2)
    RR_norm = RR_counts / (n_random * (n_random - 1) / 2)
    DR_norm = DR_counts / (n_galaxies * n_random)
    
    # Landy-Szalay estimator
    xi = (DD_norm - 2 * DR_norm + RR_norm) / RR_norm
    
    bin_midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    return bin_midpoints, xi


#main.py
def main() -> None:
    """
    Run the galaxy clustering and redshift-space distortion pipeline.
    """
    #paths can be adjusted as needed
    raw_data_path = '../data/raw_galaxies.csv' 
    cartesian_data_path = '../data/galaxies_cartesian.csv'
    
    #load and process raw galaxy data
    df_raw = load_raw_galaxy_catalog(raw_data_path)
    df_cartesian = convert_to_cartesian_coordinates(df_raw)
    df_cartesian.to_csv(cartesian_data_path, index=False)
    
    #load Cartesian coordinates for correlation computation
    df = load_cartesian_galaxy_data(cartesian_data_path)
    positions_real = df[['x', 'y', 'z_cart']].values
    
    #apply redshift space distortions
    x_rsd, y_rsd, z_rsd = apply_redshift_space_distortions(df)
    positions_rsd = np.column_stack((x_rsd, y_rsd, z_rsd))
    
    #define radial bins (in Mpc)
    bins = np.linspace(0, 150, 30)
    
    #compute correlation functions
    r_real, xi_real = compute_two_point_correlation_function(positions_real, bins)
    r_rsd, xi_rsd = compute_two_point_correlation_function(positions_rsd, bins)

if __name__ == '__main__':
    main()
