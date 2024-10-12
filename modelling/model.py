import optuna
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, MaxPooling1D
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# Load the dataset
file_path = 'transformed_cotton_production.csv'
data = pd.read_csv(file_path)

# Use only Year and Production columns, summing production by Year
yearly_production = data.groupby('Year')['Production'].sum().values.reshape(-1, 1)

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(yearly_production)

# Create dataset with look-back
def create_dataset(data, look_back=3):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 3
X, y = create_dataset(scaled_data, look_back)

# Split the dataset into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape input data to fit CNN-LSTM input shape
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the Optuna objective function
def objective(trial):
    # Hyperparameters to tune
    conv_filters = trial.suggest_int('conv_filters', 32, 128, step=32)
    lstm_units = trial.suggest_int('lstm_units', 32, 128, step=32)
    kernel_size = trial.suggest_int('kernel_size', 1, 2)  # Reduce to avoid negative dimensions
    pool_size = trial.suggest_int('pool_size', 1, 2)  # Reduce to avoid negative dimensions
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    # Build the CNN-LSTM model
    model = Sequential()
    model.add(Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu', input_shape=(look_back, 1)))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(units=lstm_units, return_sequences=True))
    model.add(LSTM(units=lstm_units))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=0)

    # Make predictions
    test_predict = model.predict(X_test)
    test_predict = scaler.inverse_transform(test_predict)

    # Evaluate model performance
    mse = mean_squared_error(y_test, test_predict)
    return mse

# Create an Optuna study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Get the best hyperparameters
best_params = study.best_params
print(f"Best hyperparameters: {best_params}")

# Train the final model with the best hyperparameters
best_model = Sequential()
best_model.add(Conv1D(filters=best_params['conv_filters'], kernel_size=best_params['kernel_size'], activation='relu', input_shape=(look_back, 1)))
best_model.add(MaxPooling1D(pool_size=best_params['pool_size']))
best_model.add(LSTM(units=best_params['lstm_units'], return_sequences=True))
best_model.add(LSTM(units=best_params['lstm_units']))
best_model.add(Dense(1))

best_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the final model
best_model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=2)

# Make predictions
final_train_predict = best_model.predict(X_train)
final_test_predict = best_model.predict(X_test)

# Inverse transform predictions
final_train_predict = scaler.inverse_transform(final_train_predict)
final_test_predict = scaler.inverse_transform(final_test_predict)
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])

# Calculate performance metrics
mse_optimized = mean_squared_error(y_test[0], final_test_predict)
mae_optimized = mean_absolute_error(y_test[0], final_test_predict)
rmse_optimized = np.sqrt(mse_optimized)

# Plot the predictions and actual values
plt.figure(figsize=(10, 6))
plt.plot(range(len(final_train_predict)), final_train_predict, label="Train Predictions", color='blue')
plt.plot(range(len(final_train_predict), len(final_train_predict) + len(final_test_predict)), final_test_predict, label="Test Predictions", color='green')
plt.plot(range(len(yearly_production)), yearly_production, label="Actual Values", color='orange')
plt.legend()
plt.title('Optimized CNN-LSTM Cotton Production Forecast')
plt.show()

# Print the performance metrics
print(f'Optimized CNN-LSTM Model MSE: {mse_optimized}')
print(f'Optimized CNN-LSTM Model MAE: {mae_optimized}')
print(f'Optimized CNN-LSTM Model RMSE: {rmse_optimized}')


# Save the trained model
joblib.dump(best_model, 'cnn_lstm_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Save the training columns (look_back)
joblib.dump(look_back, 'look_back.pkl')

# Save the entire Optuna study for future reference
joblib.dump(study, 'optuna_study.pkl')

# Save the X_train and y_train for future use
joblib.dump((X_train, y_train), 'train_data.pkl')

# Save the X_test and y_test for future evaluation
joblib.dump((X_test, y_test), 'test_data.pkl')

print("Model, scaler, and training data saved successfully.")
