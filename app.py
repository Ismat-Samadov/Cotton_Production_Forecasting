from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load('optimized_xgb_model.joblib')
scaler = joblib.load('scaler_model.joblib')  # Assuming you saved the scaler as well

# Flask App
app = Flask(__name__)

# Regions for one-hot encoding
regions = [f"Region_{r}" for r in ["Ağsu", "OtherRegion1", "OtherRegion2"]]  # List all regions here

# Home route rendering the main page
@app.route('/')
def index():
    return render_template('index.html')

# Predict API route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    year = int(data['year'])
    region = data['region']
    
    # Create a feature vector for prediction
    input_features = [year] + [1 if f'Region_{region}' == r else 0 for r in regions]
    
    # Scale the input features
    input_scaled = scaler.transform([input_features])
    
    # Make prediction
    predicted_production = model.predict(input_scaled)[0]
    
    return jsonify({"predicted_production": predicted_production})

if __name__ == '__main__':
    app.run(debug=True)
