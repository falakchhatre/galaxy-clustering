from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd

from src.data_loader import load_cartesian_galaxies
from src.rsd import apply_redshift_space_distortions
from src.correlation import compute_two_point_correlation

app = Flask(__name__)
df = load_cartesian_galaxies()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/galaxies')
def get_galaxies():
    # Params from query string
    count = int(request.args.get('count', 100))  # default 100 galaxies
    
    # Optionally filter/redshift range later
    
    # Sample galaxies if count < dataset size
    sample_df = df.sample(n=count) if count < len(df) else df
    
    # Convert to JSON-able dict
    data = sample_df[['x', 'y', 'z_cart', 'ra', 'dec', 'z']].to_dict(orient='records')
    
    return jsonify(data)

@app.route('/api/correlation', methods=['GET', 'POST'])
def get_correlation():
    if request.method == 'GET':
        # Parse parameters from query string
        count = int(request.args.get('count', 100))
        velocity_dispersion = float(request.args.get('velocity_dispersion', 300))
        
        # Sample galaxies from global df (like your /api/galaxies)
        sample_df = df.sample(n=count) if count < len(df) else df
        galaxies = sample_df.to_dict(orient='records')
        
        # Compute correlation on this data
        positions_real = np.array([[g['x'], g['y'], g['z_cart']] for g in galaxies])
        df_sample = pd.DataFrame(galaxies)
        x_rsd, y_rsd, z_rsd = apply_redshift_space_distortions(df_sample, velocity_dispersion=velocity_dispersion)
        positions_rsd = np.column_stack((x_rsd, y_rsd, z_rsd))
        
        bins = np.linspace(0, 150, 30)
        r, xi_real = compute_two_point_correlation(positions_real, bins)
        r, xi_rsd = compute_two_point_correlation(positions_rsd, bins)

        return jsonify({
            'r': r.tolist(),
            'xi_real': xi_real.tolist(),
            'xi_rsd': xi_rsd.tolist()
        })
    
    elif request.method == 'POST':
        data = request.get_json()
        velocity_dispersion = float(data.get('velocity_dispersion', 300))
        galaxies = data.get('galaxies', [])
        
        if not galaxies:
            return jsonify({'error': 'No galaxy data provided'}), 400

        positions_real = np.array([[g['x'], g['y'], g['z_cart']] for g in galaxies])
        df_sample = pd.DataFrame(galaxies)
        x_rsd, y_rsd, z_rsd = apply_redshift_space_distortions(df_sample, velocity_dispersion=velocity_dispersion)
        positions_rsd = np.column_stack((x_rsd, y_rsd, z_rsd))

        bins = np.linspace(0, 150, 30)
        r, xi_real = compute_two_point_correlation(positions_real, bins)
        r, xi_rsd = compute_two_point_correlation(positions_rsd, bins)

        return jsonify({
            'r': r.tolist(),
            'xi_real': xi_real.tolist(),
            'xi_rsd': xi_rsd.tolist(),
            'x_rsd': x_rsd.tolist(),
            'y_rsd': y_rsd.tolist(),
            'z_rsd': z_rsd.tolist()
        })


if __name__ == '__main__':
    app.run(debug=True)