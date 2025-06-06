from src.data_loader import load_raw_galaxies, add_cartesian_coords, load_cartesian_galaxies
from src.rsd import apply_redshift_space_distortions
from src.correlation import compute_two_point_correlation
import numpy as np

def main():
    df_raw = load_raw_galaxies()
    df_cart = add_cartesian_coords(df_raw)
    df_cart.to_csv('../data/galaxies_cartesian.csv', index=False)
    
    df = load_cartesian_galaxies()
    positions_real = df[['x', 'y', 'z_cart']].values
    
    x_rsd, y_rsd, z_rsd = apply_redshift_space_distortions(df)
    positions_rsd = np.column_stack((x_rsd, y_rsd, z_rsd))
    
    bins = np.linspace(0, 150, 30)
    
    r, xi_real = compute_two_point_correlation(positions_real, bins)
    r, xi_rsd = compute_two_point_correlation(positions_rsd, bins)
    
if __name__ == '__main__':
    main()
