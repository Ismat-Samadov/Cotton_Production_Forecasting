# Cotton Production Forecasting

This repository contains a machine learning project for predicting cotton production using an ARIMA model. The data source for the model is obtained from the Azerbaijani State Statistical Committee.

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

Ensure you preprocess the data into `transformed_cotton_production.csv` to use in this project.

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
2. Run the ARIMA model optimization and forecasting script:
   ```bash
   python model.py
   ```
   This script will:
   - Load the cotton production data.
   - Perform hyperparameter tuning using Optuna to find the best ARIMA parameters.
   - Train the ARIMA model on data up to 2023.
   - Forecast cotton production for the next five years (2024-2028).
   - Save the trained model and forecast results.

3. Start the Flask web application to visualize the results:
   ```bash
   python app.py
   ```
4. Open your web browser and go to `http://127.0.0.1:5000` to access the application, which includes forecasts for total cotton production, economic regions, and individual regions.

## Model Details
The model is built using the ARIMA (AutoRegressive Integrated Moving Average) time series forecasting method. Key aspects include:
- **Data Preprocessing**: 
  - The production data is aggregated by year and transformed into a time series format.
  - The production data is scaled using Min-Max scaling for consistency.
- **ARIMA Model**:
  - Hyperparameter tuning is done using Optuna to find the best ARIMA `(p, d, q)` parameters.
  - The model is trained on historical cotton production data up to 2023, and predictions are made for the next five years (2024-2028).

## Results
The ARIMA model provides accurate forecasts based on the following metrics:
- **Mean Squared Error (MSE)** for measuring prediction error.
- Forecasts are visualized in the web interface with Plotly charts, showing the next five years of cotton production.

The forecasted results for the next five years are saved to `forecasted_cotton_production.csv` and can also be viewed in the web dashboard.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.