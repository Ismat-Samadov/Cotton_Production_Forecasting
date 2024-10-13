import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = 'transformed_cotton_production.csv'
data = pd.read_csv(file_path)

# Use only Year and Production columns, summing production by Year
yearly_production = data.groupby('Year')['Production'].sum()

# Create a time series index
yearly_production.index = pd.date_range(start='2000', periods=len(yearly_production), freq='YE')

# Define look_back (set this value based on your requirement)
look_back = 5  # Example value; adjust as needed

# Create a function to evaluate the ARIMA model
def evaluate_arima(order):
    try:
        # Only use data up to the present for training
        train_data = yearly_production[:'2023']  # Up to 2023
        
        model = ARIMA(train_data, order=order)
        model_fit = model.fit()

        # Generate predictions for the next 5 years
        forecast = model_fit.forecast(steps=5)  # Forecasting for 2024-2028
        return mean_squared_error(train_data[-5:], forecast)
    except Exception as e:
        print(f"Error in model fitting: {e}")
        return np.inf  # Return a large number to indicate failure

# Define the Optuna objective function for ARIMA
def objective(trial):
    p = trial.suggest_int('p', 0, 5)  # AR term
    d = trial.suggest_int('d', 0, 2)  # Differencing
    q = trial.suggest_int('q', 0, 5)  # MA term
    
    order = (p, d, q)
    return evaluate_arima(order)

# Create an Optuna study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Get the best hyperparameters
best_params = study.best_params
print(f"Best ARIMA parameters: {best_params}")

# Train the final model with the best hyperparameters using past data up to 2023
final_model = ARIMA(yearly_production[:'2023'], order=(best_params['p'], best_params['d'], best_params['q']))
final_model_fit = final_model.fit()

# Generate future predictions for 2024-2028
forecast = final_model_fit.forecast(steps=5)  # Forecasting for the next 5 years

# Create a date range for the forecast
future_years = pd.date_range(start=yearly_production.index[-1], periods=6, freq='Y')[1:]

# Plot the predictions and actual values
plt.figure(figsize=(10, 6))
plt.plot(yearly_production.index, yearly_production.values, label="Actual Values", color='orange')
plt.plot(future_years, forecast, label="Forecast", color='green')
plt.legend()
plt.title('ARIMA Cotton Production Forecast (2024-2028)')
plt.xlabel('Year')
plt.ylabel('Production (tons)')
plt.show()

# Scaling: Create and save the scaler for future use
scaler = MinMaxScaler()

# Fit your scaler on the production data
scaler.fit(data[['Production']])

# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved successfully.")

# Save the look_back variable
joblib.dump(look_back, 'look_back.pkl')
print("Look back value saved successfully.")

# Save the trained ARIMA model
joblib.dump(final_model_fit, 'arima_model.pkl')
print("ARIMA model saved successfully.")

# Optionally, save the forecast values for future reference
forecast_df = pd.DataFrame(forecast, index=future_years, columns=['Forecast'])
forecast_df.to_csv('forecasted_cotton_production.csv')
print("Forecast values saved successfully.")

# Print performance metrics
print(f'Forecast for the next years: {forecast}')
