import streamlit as st
import pandas as pd
import numpy as np
import requests
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 60 seconds
st_autorefresh(interval=60000, limit=None, key="refresh")

# Streamlit layout
st.set_page_config(layout="wide")
st.title("📡 XAUUSD ML Signal Dashboard")

# Fetch real-time data
API_KEY = "2215ad61f67742a2a6fb9d5043777a45"
symbol = "XAU/USD"
interval = "5min"
url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY}&outputsize=500"

response = requests.get(url)
data = response.json()

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

df['ADX'] = df['close'].rolling(window=14).mean()  # Simplified
df['Volatility'] = df['close'].rolling(window=20).std()

# Label future movement
df['Target'] = np.where(df['close'].shift(-1) > df['close'], 1,
                np.where(df['close'].shift(-1) < df['close'], -1, 0))

# ML features
features = ['RSI', 'MACD', 'ADX', 'ATR', 'Volatility', 'Lag1', 'Lag2', 'EMA_Cross']
df.dropna(inplace=True)
X = df[features]
y = df['Target']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
model.fit(X_train, y_train)

# Live prediction
latest = X.iloc[-1:]
prediction = model.predict(latest)[0]
confidence = model.predict_proba(latest)[0][abs(prediction)]

# Current price and 7-day high/low
current_price = df['close'].iloc[-1]
seven_day_high = df['high'].tail(7 * 24).max()
seven_day_low = df['low'].tail(7 * 24).min()

# Display dashboard
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 ML Signal")
    st.metric("Prediction", "Buy" if prediction == 1 else "Sell" if prediction == -1 else "Hold")
    st.metric("Confidence", f"{confidence:.2f}")
    st.metric("Stop Loss", f"${df['ATR'].iloc[-1] * 1.5:.2f}")
    st.metric("Take Profit", f"${df['ATR'].iloc[-1] * 2.5:.2f}")
    st.write(f"Signal Time: {df['datetime'].iloc[-1].strftime('%I:%M %p')}")

with col2:
    st.subheader("💰 Price Overview")
    st.metric("Current Price", f"${current_price:.2f}")
    st.metric("7-Day High", f"${seven_day_high:.2f}")
    st.metric("7-Day Low", f"${seven_day_low:.2f}")
