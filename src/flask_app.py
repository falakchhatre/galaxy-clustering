from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd

from data_loader import load_cartesian_galaxies
from rsd import apply_redshift_space_distortions
from correlation import compute_two_point_correlation

app = Flask(__name__)
df = load_cartesian_galaxies()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/galaxies')
def get_galaxies():
    count = int(request.args.get('count', 100))  
    velocity_dispersion = request.args.get('velocity_dispersion', default=None, type=float)
    
    sample_df = df.sample(n=count) if count < len(df) else df

    if velocity_dispersion is not None:
        x_rsd, y_rsd, z_rsd = apply_redshift_space_distortions(sample_df, velocity_dispersion=velocity_dispersion)
        data = pd.DataFrame({
            'x': x_rsd,
            'y': y_rsd,
            'z_cart': z_rsd,
            'ra': sample_df['ra'].values,
            'dec': sample_df['dec'].values,
            'z': sample_df['z'].values
        }).to_dict(orient='records')
    else:
        data = sample_df[['x', 'y', 'z_cart', 'ra', 'dec', 'z']].to_dict(orient='records')

    return jsonify(data)

def compute_correlation_response(galaxies, velocity_dispersion):
    bins = np.linspace(0, 150, 30)
    
    positions_real = np.array([[g['x'], g['y'], g['z_cart']] for g in galaxies])
    df_sample = pd.DataFrame(galaxies)
    
    x_rsd, y_rsd, z_rsd = apply_redshift_space_distortions(df_sample, velocity_dispersion=velocity_dispersion)
    positions_rsd = np.column_stack((x_rsd, y_rsd, z_rsd))
    
    r, xi_real = compute_two_point_correlation(positions_real, bins)
    r, xi_rsd = compute_two_point_correlation(positions_rsd, bins)

    return {
        'r': r.tolist(),
        'xi_real': xi_real.tolist(),
        'xi_rsd': xi_rsd.tolist(),
        'x_rsd': x_rsd.tolist(),
        'y_rsd': y_rsd.tolist(),
        'z_rsd': z_rsd.tolist()
    }

@app.route('/api/correlation', methods=['GET', 'POST'])
def get_correlation():
    if request.method == 'GET':
        count = int(request.args.get('count', 100))
        velocity_dispersion = float(request.args.get('velocity_dispersion', 300))

        sample_df = df.sample(n=count) if count < len(df) else df
        galaxies = sample_df[['x', 'y', 'z_cart', 'ra', 'dec', 'z']].to_dict(orient='records')

        result = compute_correlation_response(galaxies, velocity_dispersion)
        return jsonify(result)

    elif request.method == 'POST':
        data = request.get_json()
        velocity_dispersion = float(data.get('velocity_dispersion', 300))
        galaxies = data.get('galaxies', [])

        if not galaxies:
            return jsonify({'error': 'No galaxy data provided'}), 400

        result = compute_correlation_response(galaxies, velocity_dispersion)
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)