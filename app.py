from flask import Flask, render_template, request, jsonify, make_response
import joblib
import logging
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the trained ARIMA model, scaler, and look-back
model = joblib.load('modelling/arima_model.pkl')  
scaler = joblib.load('modelling/scaler.pkl')  
look_back = joblib.load('modelling/look_back.pkl')

# Load the cotton production data
file_path = 'modelling/transformed_cotton_production.csv'
df = pd.read_csv(file_path)

# Flask App
app = Flask(__name__)

# Home route rendering the main page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    year = int(request.args.get('year'))  # Get the year from the query parameter
    logging.info(f"Received year: {year}")

    # Check if the year has enough historical data
    if year < (2000 + look_back):
        return make_response(jsonify({"error": "Insufficient data for the requested year"}), 400)

    # Generate prediction using the ARIMA model
    try:
        forecast = model.get_forecast(steps=5)  # Predict the next 5 years
        predicted_values = forecast.predicted_mean.tolist()  # Convert the forecast to a list
        logging.info(f"Forecast values: {predicted_values}")
        
        response = jsonify({"predictions": predicted_values})
        return make_response(response, 200)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return make_response(jsonify({"error": str(e)}), 500)

if __name__ == '__main__':
    app.run(debug=True)
