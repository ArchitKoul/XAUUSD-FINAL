import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import time
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 60 seconds
st_autorefresh(interval=60000, limit=None, key="refresh")

# Streamlit layout
st.set_page_config(layout="wide")
st.title("📉 XAUUSD ML Signal Dashboard")

# Sidebar controls
st.sidebar.title("🧠 Model Selector")
model_choice = st.sidebar.selectbox("Choose ML Model", ["XGBoost", "Logistic Regression"], key="model_selector")

st.sidebar.title("⚙️ Retrain Settings")
retrain_interval = st.sidebar.selectbox("Retrain every...", ["Every refresh", "5 minutes", "15 minutes", "1 hour"], key="retrain_selector")

use_price_action = st.sidebar.checkbox("Enable Price Action Strategy")

# Retrain frequency logic
if "last_retrain" not in st.session_state:
    st.session_state.last_retrain = 0

interval_map = {
    "Every refresh": 0,
    "5 minutes": 300,
    "15 minutes": 900,
    "1 hour": 3600
}
should_retrain = (time.time() - st.session_state.last_retrain) > interval_map[retrain_interval]

# Fetch real-time data
API_KEY = "2215ad61f67742a2a6fb9d5043777a45"
symbol = "XAU/USD"
interval = "5min"
url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY}&outputsize=500"

response = requests.get(url)
data = response.json()

if 'values' not in data:
    st.error("❌ API Error: No data returned. Check your API key, symbol, or usage limits.")
    st.stop()

df = pd.DataFrame(data['values'])
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')
df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)

# Technical indicators
df['Return'] = np.log(df['close'] / df['close'].shift(1))
df['Lag1'] = df['Return'].shift(1)
df['Lag2'] = df['Return'].shift(2)
df['EMA_20'] = df['close'].ewm(span=20).mean()
df['EMA_50'] = df['close'].ewm(span=50).mean()
df['EMA_Cross'] = (df['EMA_20'] > df['EMA_50']).astype(int)

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

df['ADX'] = df['close'].rolling(window=14).mean()
df['Volatility'] = df['close'].rolling(window=20).std()

df['Target'] = np.where(df['close'].shift(-1) > df['close'], 2,
                np.where(df['close'].shift(-1) < df['close'], 0, 1))

# Price action features
df['Body'] = np.abs(df['close'] - df['open'])
df['Upper_Wick'] = df['high'] - df[['close', 'open']].max(axis=1)
df['Lower_Wick'] = df[['close', 'open']].min(axis=1) - df['low']
df['Bullish_Engulfing'] = ((df['close'] > df['open']) & 
                           (df['close'].shift(1) < df['open'].shift(1)) &
                           (df['close'] > df['open'].shift(1)) &
                           (df['open'] < df['close'].shift(1))).astype(int)
df['Bearish_Engulfing'] = ((df['close'] < df['open']) & 
                           (df['close'].shift(1) > df['open'].shift(1)) &
                           (df['close'] < df['open'].shift(1)) &
                           (df['open'] > df['close'].shift(1))).astype(int)
df['Price_Action_Signal'] = (
    (df['close'] > df['high'].shift(1)) & 
    (df['Bullish_Engulfing'] == 1)
).astype(int)

features = ['RSI', 'MACD', 'ADX', 'ATR', 'Volatility', 'Lag1', 'Lag2', 'EMA_Cross']
df.dropna(inplace=True)

# Walk-forward retraining
window_size = 300
df = df.tail(window_size + 1)

train_df = df.iloc[:-1]
test_df = df.iloc[-1:]

X_train = train_df[features]
y_train = train_df['Target']
X_test = test_df[features].to_frame() if isinstance(test_df[features], pd.Series) else test_df[features]

if should_retrain or "model" not in st.session_state:
    if model_choice == "XGBoost":
        model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    st.session_state.model = model
    st.session_state.last_retrain = time.time()
else:
    model = st.session_state.model

# Signal selection
if use_price_action:
    df['Signal'] = df['Price_Action_Signal'].replace({1: 2, 0: 1})
    latest_signal = df['Signal'].iloc[-1]
    latest_confidence = 1.0
else:
    X = df[features]
    df['Signal'] = model.predict(X)
    latest_signal = df['Signal'].iloc[-1]
    latest_confidence = model.predict_proba(X_test)[0][latest_signal]

# Price overview
current_price = df['close'].iloc[-1]
seven_day_high = df['high'].tail(7 * 24).max()
seven_day_low = df['low'].tail(7 * 24).min()

col1, col2 = st.columns(2)
with col1:
    st.subheader("⚔️ Signal")
    direction_map = {0: "Sell", 1: "Hold", 2: "Buy"}
    st.metric("Prediction", direction_map[latest_signal])
    st.metric("Confidence", f"{latest_confidence:.2f}")
    st.metric("Stop Loss", f"${df['ATR'].iloc[-1] * 1.5:.2f}")
    st.metric("Take Profit", f"${df['ATR'].iloc[-1] * 2.5:.2f}")
    st.write(f"Signal Time: {df['datetime'].iloc[-1].strftime('%I:%M %p')}")

with col2:
    st.subheader("💰 Price Overview")
    st.metric("Current Price", f"${current_price:.2f}")
    st.metric("7-Day High", f"${seven_day_high:.2f}")
    st.metric("7-Day Low", f"${seven_day_low:.2f}")

# 🌐 Macro Overlay (static for now)
st.subheader("🌐 Macro Overlay")
macro_col1, macro_col2, macro_col3 = st.columns(3)
macro_col1.metric("DXY (Dollar Index)", "106.12")
macro_col2.metric("US CPI YoY", "3.7%")
macro_col3.metric("Fed Funds Rate", "5.50%")
st.caption("Next macro event: US CPI release on Oct 10, 2025")

# Strategy simulation
df['Position'] = df['Signal'].replace({0: -1, 1: 0, 2: 1})
df['Market_Return'] = df
