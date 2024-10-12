from flask import Flask, render_template, request, jsonify, make_response
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import logging
import numpy as np
# Set up logging
logging.basicConfig(level=logging.INFO)

# Use Agg backend for matplotlib to avoid GUI issues
plt.switch_backend('Agg')

# Load the unique regions from the dataset
file_path = 'modelling/transformed_cotton_production.csv'
df = pd.read_csv(file_path)
unique_regions = df['Region'].unique()

feature_names = ['Year', 'Region']  # List your features used in training
joblib.dump(feature_names, 'modelling/feature_names.joblib')


# Load the trained model, scaler, and feature names
model = joblib.load('modelling/cnn_lstm_model.pkl')  
scaler = joblib.load('modelling/scaler.pkl')  # Load the MinMaxScaler
look_back = joblib.load('modelling/look_back.pkl')  # Load the look-back window

# Flask App
app = Flask(__name__)

# Home route rendering the main page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_all', methods=['POST'])
def predict_all():
    data = request.get_json()
    year = int(data['year'])  # Ensure year is correctly parsed

    # Log the year for debugging
    logging.info(f"Received year: {year}")

    predictions = {}

    # Prepare input for each region
    for region in unique_regions:
        # Create input features: year + one-hot encoding for the region
        input_features = [year]  # Use the year directly, without scaling

        # One-hot encode the region
        for feature in feature_names[1:]:  # Skip the first feature, 'Year'
            if feature == f'Region_{region}':
                input_features.append(1)  # One-hot encoding
            else:
                input_features.append(0)

        # Ensure only the numerical part is scaled (not the one-hot region encoding)
        year_scaled = scaler.transform([[year]])  # Scale the year only
        input_features[0] = year_scaled[0][0]  # Replace the year with the scaled year

        # Convert input_features to a NumPy array
        input_scaled = np.array([input_features])  # Convert to 2D array

        # Perform the prediction
        predicted_production = model.predict(input_scaled)[0]
        predictions[region] = float(predicted_production)  # Convert float32 to float

    # Sort predictions alphabetically by region name
    sorted_predictions = dict(sorted(predictions.items(), key=lambda item: item[0]))

    # Generate a bar chart for predictions
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_predictions.keys(), sorted_predictions.values(), color='blue')
    plt.xticks(rotation=90)
    plt.title(f'Predicted Cotton Production for All Regions in {year}')
    plt.xlabel('Region')
    plt.ylabel('Predicted Production')

    # Save the plot to a bytes object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_base64 = base64.b64encode(img.read()).decode('utf-8')
    plt.close()

    # Convert predictions to a list of dictionaries for easy frontend consumption
    predictions_list = [{"region": region, "production": prediction} for region, prediction in sorted_predictions.items()]

    # Disable caching of API responses
    response = make_response(jsonify({
        "chart": chart_base64,
        "predictions": predictions_list  # Return predictions
    }))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'

    return response


if __name__ == '__main__':
    app.run(debug=True)
