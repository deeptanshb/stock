import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Multi-Stock Forecasting", layout="wide")
st.title("📈 Stock Price Forecasting for Multiple Tickers")

tickers = st.multiselect(
    "Select up to 5 Stock Tickers (e.g., AAPL, MSFT, TSLA)",
    ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    default=["AAPL"],
    max_selections=5
)
model_choice = st.selectbox("Choose Forecasting Model", ["ARIMA", "SARIMA", "Prophet", "LSTM"])
start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2025-05-08"))

@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    return df.dropna()

def forecast_arima(df):
    df.set_index('ds', inplace=True)
    model = ARIMA(df['y'], order=(5,1,0))
    result = model.fit()
    forecast = result.forecast(steps=365)
    fig, ax = plt.subplots()
    df['y'].plot(ax=ax, label='Historical')
    forecast.plot(ax=ax, label='ARIMA Forecast')
    plt.legend()
    plt.title("ARIMA Forecast")
    return fig

def forecast_sarima(df):
    df.set_index('ds', inplace=True)
    model = SARIMAX(df['y'], order=(1,1,1), seasonal_order=(1,1,1,12))
    result = model.fit()
    forecast = result.forecast(steps=365)
    fig, ax = plt.subplots()
    df['y'].plot(ax=ax, label='Historical')
    forecast.plot(ax=ax, label='SARIMA Forecast')
    plt.legend()
    plt.title("SARIMA Forecast")
    return fig

def forecast_prophet(data):
    data.columns = data.columns.get_level_values(0)
    df_prophet = data.reset_index()
    df_prophet = df_prophet.rename(columns={'your_date_column_name': 'ds', 'your_value_column_name': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
    df_prophet.dropna(subset=['ds', 'y'], inplace=True)
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_prophet['ds'], df_prophet['y'], 'k.', label='Actual')
    ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecast (yhat)')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                    color='skyblue', alpha=0.3, label='Confidence Interval')
    ax.set_title('Prophet Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

def forecast_lstm(df):
    df_lstm = df.copy()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_lstm[['y']])

    def create_sequences(data, seq_len=60):
        x, y = [], []
        for i in range(seq_len, len(data)):
            x.append(data[i-seq_len:i])
            y.append(data[i])
        return np.array(x), np.array(y)

    seq_len = 60
    x, y_seq = create_sequences(scaled, seq_len)
    split = int(0.8 * len(x))
    x_train, y_train = x[:split], y_seq[:split]
    x_test, y_test = x[split:], y_seq[split:]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_len, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

    pred = model.predict(x_test)
    pred_inv = scaler.inverse_transform(pred)
    y_test_inv = scaler.inverse_transform(y_test)

    fig, ax = plt.subplots()
    ax.plot(y_test_inv, label='Actual')
    ax.plot(pred_inv, label='LSTM Forecast')
    plt.legend()
    plt.title("LSTM Forecast")
    return fig

for ticker in tickers:
    st.subheader(f"📊 {ticker} — {model_choice} Forecast")
    df = load_data(ticker, start_date, end_date)
    st.line_chart(df.set_index('ds')['y'], use_container_width=True)

    if model_choice == "ARIMA":
        fig = forecast_arima(df.copy())
    elif model_choice == "SARIMA":
        fig = forecast_sarima(df.copy())
    elif model_choice == "Prophet":
        fig = forecast_prophet(df.copy())
    elif model_choice == "LSTM":
        fig = forecast_lstm(df.copy())

    st.pyplot(fig)
    st.markdown("---")
