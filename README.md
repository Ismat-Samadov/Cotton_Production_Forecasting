# Cotton Production Forecasting

This repository contains a machine learning project for predicting cotton production using a CNN-LSTM model. The data source for the model is obtained from the Azerbaijani State Statistical Committee.

## Table of Contents
- [Data Source](#data-source)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Data Source
The data used for this model can be found at the following link:
- [Agricultural Statistics](https://www.stat.gov.az/source/agriculture/az/2.128.xls)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Ismat-Samadov/Cotton_Production_Forecasting.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Cotton_Production_Forecasting
   ```
3. Set up a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\activate
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Ensure you have the necessary data file `transformed_cotton_production.csv` in the project directory.
2. Run the model:
   ```bash
   python model.py
   ```
3. Start the Flask web application:
   ```bash
   python app.py
   ```
4. Open your web browser and go to `http://127.0.0.1:5000` to access the application.

## Model Details
The model is built using a Convolutional Neural Network (CNN) combined with a Long Short-Term Memory (LSTM) architecture. Key aspects include:
- **Data Preprocessing**: The production data is scaled using Min-Max scaling.
- **Model Architecture**: 
  - Convolutional layers followed by LSTM layers for capturing temporal dependencies.
  - Hyperparameter tuning using Optuna for optimal model performance.

## Results
The performance metrics for the optimized model include:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Predictions for cotton production are provided in the web application interface.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.