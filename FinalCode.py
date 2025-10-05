import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 60 seconds
st_autorefresh(interval=60000, limit=None, key="refresh")

# Streamlit layout
st.set_page_config(layout="wide")
st.title("📡 XAUUSD Live Signal Dashboard")

# Fetch real-time data from Twelve Data
API_KEY = "2215ad61f67742a2a6fb9d5043777a45"  # Replace with your own key if needed
symbol = "XAU/USD"
interval = "5min"
url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY}&outputsize=30"

response = requests.get(url)
data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data['values'])
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')
df['close'] = df['close'].astype(float)
df['high'] = df['high'].astype(float)
df['low'] = df['low'].astype(float)

# Calculate technical indicators
def calculate_technical_indicators(df):
    delta = df['close'].diff()
    gain = delta.mask(delta < 0, 0)
    loss = delta.mask(delta > 0, 0).abs()
    avg_gain = gain.ewm(com=14, min_periods=14).mean()
    avg_loss = loss.ewm(com=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(com=14, min_periods=14).mean()

    df['ADX'] = df['close'].rolling(window=14).mean()  # Simplified placeholder
    df['Volatility'] = df['close'].rolling(window=20).std()

    return df

df = calculate_technical_indicators(df)

# Generate signal
latest = df.iloc[-1]
confidence_score = sum([
    latest['RSI'] < 30,
    latest['MACD'] > 0,
    latest['ADX'] > 25,
    latest['Volatility'] < df['Volatility'].rolling(20).mean().iloc[-1]
])

signal = {
    "Direction": "Buy" if confidence_score >= 3 else "Hold",
    "Confidence": confidence_score,
    "SL": latest['ATR'] * 1.5,
    "TP": latest['ATR'] * 2.5,
    "Time": latest['datetime'].strftime("%H:%M %p")
}

# Display signal panel
st.subheader("📈 Live Signal")
st.metric("Direction", signal["Direction"])
st.metric("Confidence Score", f"{signal['Confidence']}/4")
st.metric("Stop Loss", f"${signal['SL']:.2f}")
st.metric("Take Profit", f"${signal['TP']:.2f}")
st.write(f"Signal Time: {signal['Time']}")

import plotly.graph_objects as go

fig = go.Figure(data=[go.Candlestick(
    x=df['datetime'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close']
)])
fig.update_layout(title="XAUUSD Candlestick", xaxis_title="Time", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)
