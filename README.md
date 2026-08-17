# Time Series Forecasting Dashboard

## Project Overview

This project performs time series analysis and forecasting on monthly sales data.

The project uses statistical techniques and an ARIMA model to analyze trends, test stationarity, and forecast future sales.

## Objective

The main objective is to:

- Analyze temporal sales data
- Resample time series data
- Calculate rolling statistics
- Check stationarity using the ADF test
- Apply differencing
- Analyze ACF and PACF
- Perform time series decomposition
- Train an ARIMA forecasting model
- Forecast future sales
- Evaluate model performance
- Build an interactive Streamlit dashboard

## Dataset

The dataset contains monthly sales records from:

**January 2021 to December 2025**

Columns:

- Date
- Sales

Total records:

**60**

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Statsmodels
- Scikit-learn
- Joblib
- Streamlit

## Time Series Analysis

### Resampling

The sales data is converted into monthly time series data using Pandas resampling.

### Rolling Statistics

A 7-month rolling mean is calculated to analyze the underlying trend.

### Stationarity Test

The Augmented Dickey-Fuller (ADF) test is used to check whether the time series is stationary.

The original series was not stationary.

First-order differencing was also not stationary.

Second-order differencing made the series stationary.

Therefore:

**ARIMA d = 2**

## ARIMA Model

The ARIMA model used in this project is:

**ARIMA(1, 2, 1)**

Where:

- p = 1
- d = 2
- q = 1

The trained model is saved using Joblib:

```text
models/arima_model.pkl