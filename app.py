from flask import Flask, render_template, request, jsonify, make_response
import joblib
import logging
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the trained model, scaler, and look-back
model = joblib.load('modelling/cnn_lstm_model.pkl')  
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

    # We need a sequence of years for the input based on the look_back window
    # Check if the year has enough historical data
    if year < (2000 + look_back):
        return make_response(jsonify({"error": "Insufficient data for the requested year"}), 400)

    # Prepare the input data
    past_years = np.array([[year - i] for i in range(look_back, 0, -1)])

    # Log the past years used for the input
    logging.info(f"Past years input: {past_years}")

    # Scale the past years
    try:
        scaled_past_years = scaler.transform(past_years)
        logging.info(f"Scaled past years input: {scaled_past_years}")
    except Exception as e:
        logging.error(f"Error in scaling input: {e}")
        return make_response(jsonify({"error": "Failed to scale input data"}), 500)

    # Reshape the input to fit the model's expected input shape (1, look_back, 1)
    input_reshaped = scaled_past_years.reshape((1, look_back, 1))

    # Generate prediction using the model
    try:
        prediction = model.predict(input_reshaped)[0][0]  # Get the predicted production
        prediction = scaler.inverse_transform([[prediction]])[0][0]  # Inverse scale to get actual production
        logging.info(f"Predicted production (actual value): {prediction}")  # Log the actual value
        # response = jsonify({"prediction": f"Predicted Production for {year}: {round(prediction, 2)} tons"})
        response = jsonify({"prediction": prediction})  # Directly return the float value
        return make_response(response, 200)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return make_response(jsonify({"error": str(e)}), 500)


    
if __name__ == '__main__':
    app.run(debug=True)
