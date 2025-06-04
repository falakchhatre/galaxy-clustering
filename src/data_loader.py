import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import Planck18
import os

BASE_PATH = os.path.dirname(__file__)
DEFAULT_CARTESIAN_PATH = os.path.abspath(os.path.join(BASE_PATH, '..', 'data', 'galaxies_cartesian.csv'))

def load_raw_galaxies(path=DEFAULT_CARTESIAN_PATH):
    df = pd.read_csv(path)
    if 'zWarning' in df.columns:
        df = df.drop(columns=['zWarning'])
    return df

def add_cartesian_coords(df):
    coords = SkyCoord(ra=df['ra'].values * u.degree,
                      dec=df['dec'].values * u.degree,
                      frame='icrs')
    distances = Planck18.comoving_distance(df['z'].values)
    
    ra_rad = coords.ra.radian
    dec_rad = coords.dec.radian
    r = distances.value
    
    df['x'] = r * np.cos(dec_rad) * np.cos(ra_rad)
    df['y'] = r * np.cos(dec_rad) * np.sin(ra_rad)
    df['z_cart'] = r * np.sin(dec_rad)
    return df

def load_cartesian_galaxies(path=DEFAULT_CARTESIAN_PATH):
    return pd.read_csv(path)
