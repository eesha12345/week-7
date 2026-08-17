# STEP 13 - Train and Save ARIMA Model

import pandas as pd
import joblib

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load dataset
df = pd.read_csv("dataset.csv")

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Sort data by date
df = df.sort_values("Date")

# Set Date as index
df = df.set_index("Date")

# Resample to monthly frequency
monthly_sales = df["Sales"].resample("MS").sum()

# Split data
train_size = int(len(monthly_sales) * 0.8)

train = monthly_sales[:train_size]
test = monthly_sales[train_size:]

print("Training Size:", len(train))
print("Testing Size:", len(test))

# Create ARIMA model
model = ARIMA(train, order=(1, 2, 1))

# Train model
model_fit = model.fit()

# Forecast test data
forecast = model_fit.forecast(steps=len(test))

# Calculate errors
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
mape = np.mean(np.abs((test - forecast) / test)) * 100

print("\nModel Performance")
print("----------------------------")
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape, "%")

# Save model
joblib.dump(model_fit, "models/arima_model.pkl")

print("\nARIMA model saved successfully!")
print("Location: models/arima_model.pkl")