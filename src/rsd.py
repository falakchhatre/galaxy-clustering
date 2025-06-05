import numpy as np
from astropy.coordinates import SkyCoord
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

    coords = SkyCoord(ra=df['ra'].values * u.deg,
                      dec=df['dec'].values * u.deg,
                      distance=dist_rsd * u.Mpc)

    x_rsd = coords.cartesian.x.value
    y_rsd = coords.cartesian.y.value
    z_rsd = coords.cartesian.z.value

    return x_rsd, y_rsd, z_rsd
