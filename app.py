import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error


# PAGE SETTINGS
st.set_page_config(
    page_title="Time Series Forecasting",
    page_icon="📈",
    layout="wide"
)


# LOAD DATA
df = pd.read_csv("dataset.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df = df.set_index("Date")

monthly_sales = df["Sales"].resample("MS").sum()


# LOAD ARIMA MODEL
model = joblib.load("arima_model.pkl")


# SIDEBAR
st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Dataset",
        "Trend Analysis",
        "Decomposition",
        "ACF & PACF",
        "Forecast",
        "Model Performance"
    ]
)


# HOME
if page == "Home":

    st.title("📈 Time Series Forecasting Dashboard")

    st.write(
        "This project analyzes monthly sales data and uses "
        "ARIMA to forecast future sales."
    )

    st.subheader("Project Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(monthly_sales))

    with col2:
        st.metric(
            "Start Date",
            monthly_sales.index.min().strftime("%Y-%m")
        )

    with col3:
        st.metric(
            "End Date",
            monthly_sales.index.max().strftime("%Y-%m")
        )

    st.subheader("Machine Learning Workflow")

    st.write("""
    1. Load and clean time series data
    2. Resample data
    3. Calculate rolling statistics
    4. Perform ADF stationarity test
    5. Apply differencing
    6. Perform time series decomposition
    7. Analyze ACF and PACF
    8. Train ARIMA model
    9. Forecast future values
    10. Evaluate model performance
    """)


# DATASET
elif page == "Dataset":

    st.title("📋 Dataset")

    st.write("Monthly sales data used for forecasting.")

    st.dataframe(
        monthly_sales.reset_index(),
        use_container_width=True
    )

    st.subheader("Dataset Statistics")

    st.dataframe(
        monthly_sales.describe().to_frame(),
        use_container_width=True
    )


# TREND ANALYSIS
elif page == "Trend Analysis":

    st.title("📈 Trend Analysis")

    rolling_mean = monthly_sales.rolling(
        window=7
    ).mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=monthly_sales.index,
            y=monthly_sales,
            mode="lines+markers",
            name="Actual Sales"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=rolling_mean.index,
            y=rolling_mean,
            mode="lines",
            name="7-Month Rolling Mean"
        )
    )

    fig.update_layout(
        title="Monthly Sales and Rolling Mean",
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# DECOMPOSITION
elif page == "Decomposition":

    st.title("🔍 Time Series Decomposition")

    st.write(
        "Trend, seasonal and residual components."
    )

    st.image(
        "decomposition.png"
    )


# ACF AND PACF
elif page == "ACF & PACF":

    st.title("📊 ACF & PACF Analysis")

    st.write(
        "Autocorrelation and Partial Autocorrelation "
        "analysis of the sales time series."
    )

    # First-order differencing
    differenced_sales = monthly_sales.diff().dropna()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("ACF Plot")

        fig_acf, ax_acf = plt.subplots(
            figsize=(8, 4)
        )

        plot_acf(
            differenced_sales,
            lags=20,
            ax=ax_acf
        )

        ax_acf.set_title(
            "Autocorrelation Function (ACF)"
        )

        st.pyplot(fig_acf)

        plt.close(fig_acf)

    with col2:

        st.subheader("PACF Plot")

        fig_pacf, ax_pacf = plt.subplots(
            figsize=(8, 4)
        )

        plot_pacf(
            differenced_sales,
            lags=20,
            ax=ax_pacf,
            method="ywm"
        )

        ax_pacf.set_title(
            "Partial Autocorrelation Function (PACF)"
        )

        st.pyplot(fig_pacf)

        plt.close(fig_pacf)


# FORECAST
elif page == "Forecast":

    st.title("🔮 Sales Forecast")

    forecast_months = st.slider(
        "Number of months to forecast",
        1,
        24,
        12
    )

    forecast = model.forecast(
        steps=forecast_months
    )

    future_dates = pd.date_range(
        start=monthly_sales.index[-1]
        + pd.DateOffset(months=1),
        periods=forecast_months,
        freq="MS"
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast.values
    })

    st.subheader("Forecasted Values")

    st.dataframe(
        forecast_df,
        use_container_width=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=monthly_sales.index,
            y=monthly_sales,
            mode="lines",
            name="Historical Sales"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=forecast.values,
            mode="lines+markers",
            name="Forecast"
        )
    )

    fig.update_layout(
        title="Historical Sales and Future Forecast",
        xaxis_title="Date",
        yaxis_title="Sales"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    csv = forecast_df.to_csv(index=False)

    st.download_button(
        "📥 Download Forecast CSV",
        csv,
        "sales_forecast.csv",
        "text/csv"
    )


# MODEL PERFORMANCE
elif page == "Model Performance":

    st.title("📊 Model Performance")

    train_size = int(
        len(monthly_sales) * 0.8
    )

    test = monthly_sales[train_size:]

    test_forecast = model.forecast(
        steps=len(test)
    )

    mae = mean_absolute_error(
        test,
        test_forecast
    )

    rmse = np.sqrt(
        mean_squared_error(
            test,
            test_forecast
        )
    )

    mape = np.mean(
        np.abs(
            (test - test_forecast) / test
        )
    ) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "MAE",
            f"{mae:.2f}"
        )

    with col2:
        st.metric(
            "RMSE",
            f"{rmse:.2f}"
        )

    with col3:
        st.metric(
            "MAPE",
            f"{mape:.2f}%"
        )

    st.subheader("Actual vs Forecast")

    comparison = pd.DataFrame({
        "Actual": test,
        "Forecast": test_forecast
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=test,
            mode="lines+markers",
            name="Actual"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=test_forecast,
            mode="lines+markers",
            name="Forecast"
        )
    )

    fig.update_layout(
        title="Actual vs ARIMA Forecast",
        xaxis_title="Date",
        yaxis_title="Sales"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
