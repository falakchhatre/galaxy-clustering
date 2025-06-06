import numpy as np
from astropy.coordinates import spherical_to_cartesian
import astropy.units as u
from astropy.cosmology import Planck18

def apply_redshift_space_distortions(df, velocity_dispersion=300):
    """
    Add line-of-sight peculiar velocity shifts to galaxy redshifts
    and compute distorted Cartesian coordinates.
    """
    c = 3e5
    v_peculiar = np.random.normal(0, velocity_dispersion, len(df))
    delta_z = (v_peculiar / c) * (1 + df['z'].values)
    z_obs = df['z'].values + delta_z

    if np.any(z_obs < 0):
        raise ValueError("Negative observed redshift found!")

    dist_rsd = Planck18.comoving_distance(z_obs).value
    ra_rad = np.deg2rad(df['ra'].values)
    dec_rad = np.deg2rad(df['dec'].values)

    x, y, z = spherical_to_cartesian(dist_rsd, dec_rad, ra_rad)
    return x.value, y.value, z.value