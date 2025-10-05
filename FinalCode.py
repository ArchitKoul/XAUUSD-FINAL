import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt

# Auto-refresh every 60 seconds
st_autorefresh(interval=60000, limit=None, key="refresh")
st.set_page_config(layout="wide")
st.title("📉 XAUUSD ML Signal Dashboard")

# Sidebar controls
model_choice = st.sidebar.selectbox("Choose ML Model", ["XGBoost", "Logistic Regression"])
retrain_interval = st.sidebar.selectbox("Retrain every...", ["Every refresh", "5 minutes", "15 minutes", "1 hour"])
use_price_action = st.sidebar.checkbox("Enable Price Action Strategy")
chart_interval = st.sidebar.selectbox("Chart timeframe", ["15min", "30min", "4h", "1day"])

# Retrain logic
if "last_retrain" not in st.session_state:
    st.session_state.last_retrain = 0
interval_map = {"Every refresh": 0, "5 minutes": 300, "15 minutes": 900, "1 hour": 3600}
should_retrain = (time.time() - st.session_state.last_retrain) > interval_map[retrain_interval]

# Fetch real-time data
API_KEY = "2215ad61f67742a2a6fb9d5043777a45"
symbol = "XAU/USD"
url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=5min&apikey={API_KEY}&outputsize=500"
response = requests.get(url).json()
if 'values' not in response:
    st.error("API error")
    st.stop()

df = pd.DataFrame(response['values'])
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
avg_gain = gain.ewm(com=14).mean()
avg_loss = loss.ewm(com=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))
exp1 = df['close'].ewm(span=12).mean()
exp2 = df['close'].ewm(span=26).mean()
df['MACD'] = exp1 - exp2
df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
tr = pd.concat([
    df['high'] - df['low'],
    abs(df['high'] - df['close'].shift()),
    abs(df['low'] - df['close'].shift())
], axis=1).max(axis=1)
df['ATR'] = tr.ewm(span=14).mean()
df['ADX'] = df['close'].rolling(window=14).mean()
df['Volatility'] = df['close'].rolling(window=20).std()
df['Target'] = np.where(df['close'].shift(-1) > df['close'], 2,
                np.where(df['close'].shift(-1) < df['close'], 0, 1))

# Candlestick pattern detection
df['Body'] = abs(df['close'] - df['open'])
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
df['Hammer'] = ((df['Body'] < df['ATR']) &
                (df['Lower_Wick'] > df['Body'] * 2) &
                (df['Upper_Wick'] < df['Body'])).astype(int)
df['Inverted_Hammer'] = ((df['Body'] < df['ATR']) &
                         (df['Upper_Wick'] > df['Body'] * 2) &
                         (df['Lower_Wick'] < df['Body'])).astype(int)
df['Reversal'] = ((df['Lower_Wick'] > df['Body'] * 1.5) &
                  (df['close'] > df['open'])).astype(int)
df['Bearish_Reversal'] = ((df['Upper_Wick'] > df['Body'] * 1.5) &
                          (df['close'] < df['open'])).astype(int)
df.dropna(inplace=True)

# Feature set
features = ['RSI', 'MACD', 'ADX', 'ATR', 'Volatility', 'Lag1', 'Lag2', 'EMA_Cross',
            'Bullish_Engulfing', 'Bearish_Engulfing', 'Hammer', 'Inverted_Hammer',
            'Reversal', 'Bearish_Reversal']

# Walk-forward training
window_size = 300
df = df.tail(window_size + 1)
train_df = df.iloc[:-1]
test_df = df.iloc[-1:]
X_train = train_df[features]
y_train = train_df['Target']
X_test = test_df[features].to_frame() if isinstance(test_df[features], pd.Series) else test_df[features]

if should_retrain or "model" not in st.session_state:
    model = XGBClassifier() if model_choice == "XGBoost" else LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    st.session_state.model = model
    st.session_state.last_retrain = time.time()
else:
    model = st.session_state.model

# Signal selection
if use_price_action:
    df['Signal'] = df['Bullish_Engulfing'].replace({1: 2, 0: 1})
    df['Confidence'] = (df['Body'] / df['ATR']).clip(0, 1)
    latest_confidence = df['Confidence'].iloc[-1]
else:
    df['Signal'] = model.predict(df[features])
    df['Confidence'] = model.predict_proba(df[features])[np.arange(len(df)), df['Signal']]
    latest_confidence = model.predict_proba(X_test)[0][df['Signal'].iloc[-1]]

latest_signal = df['Signal'].iloc[-1]
direction_map = {0: "Sell", 1: "Hold", 2: "Buy"}

# Display signal
st.subheader("⚔️ Signal")
st.metric("Prediction", direction_map[latest_signal])
st.metric("Confidence", f"{latest_confidence:.2f}")
st.metric("Stop Loss", f"${df['ATR'].iloc[-1] * 1.5:.2f}")
st.metric("Take Profit", f"${df['ATR'].iloc[-1] * 2.5:.2f}")
st.write(f"Signal Time: {df['datetime'].iloc[-1].strftime('%I:%M %p')}")

# Display current pattern
pattern_map = {
    'Bullish_Engulfing': "Bullish Engulfing",
    'Bearish_Engulfing': "Bearish Engulfing",
    'Hammer': "Hammer",
    'Inverted_Hammer': "Inverted Hammer",
    'Reversal': "Bullish Reversal",
    'Bearish_Reversal': "Bearish Reversal"
}
current_patterns = [name for name in pattern_map if df[name].iloc[-1] == 1]
if current_patterns:
    st.subheader("📐 Current Candlestick Pattern")
    st.success(f"Pattern forming: {', '.join([pattern_map[p] for p in current_patterns])}")
else:
    st.subheader("📐 Current Candlestick Pattern")
    st.info("No clear pattern forming")

# Candlestick chart with timeframe filter
chart_url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval={chart_interval}&apikey={API_KEY}&outputsize=100"
chart_data = requests.get(chart_url).json()
if 'values' in chart_data:
    chart_df = pd.DataFrame(chart_data['values'])
    chart_df['datetime'] = pd.to_datetime(chart_df['datetime'])
    chart_df = chart_df.sort_values('datetime')
    chart_df[['open', 'high', 'low', 'close']] = chart_df[['open', 'high', 'low', 'close']].astype(float)

    fig = go.Figure(data=[go.Candlestick(
        x=chart_df['datetime'],
        open=chart_df['open'],
        high=chart_df['high'],
        low=chart_df['low'],
        close=chart_df['close']
    )])
    fig.update_layout(title=f"XAUUSD Candlestick Chart ({chart_interval})", xaxis_rangeslider_visible=False)
    st.subheader("📈 XAUUSD Chart")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Unable to load chart data.")

# Strategy simulation
df['Position'] = df['Signal'].replace({0: -1, 1: 0, 2: 1})
df['Market_Return'] = df['Return']
df['Strategy_Return'] = df['Position'].shift(1) * df['Market_Return']
df['Cumulative_Market'] = (1 + df['Market_Return']).cumprod()
df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()

# Strategy performance metrics
win_trades = df[df['Strategy_Return'] > 0].shape[0]
loss_trades = df[df['Strategy_Return'] < 0].shape[0]
win_rate = win_trades / (win_trades + loss_trades) if (win_trades + loss_trades) > 0 else 0
avg_gain = df[df['Strategy_Return'] > 0]['Strategy_Return'].mean()
avg_loss = df[df['Strategy_Return'] < 0]['Strategy_Return'].mean()
sharpe = df['Strategy_Return'].mean() / df['Strategy_Return'].std() * np.sqrt(252)

# Plot cumulative performance
st.subheader("📊 Strategy Performance")
fig_perf, ax = plt.subplots(figsize=(10, 4))
ax.plot(df['datetime'], df['Cumulative_Market'], label='Market', color='gray')
ax.plot(df['datetime'], df['Cumulative_Strategy'], label='Strategy', color='blue')
ax.set_title("Cumulative Returns")
ax.legend()
st.pyplot(fig_perf)

# Display metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Win Rate", f"{win_rate:.2%}")
col2.metric("Avg Gain", f"{avg_gain:.4f}")
col3.metric("Avg Loss", f"{avg_loss:.4f}")
col4.metric("Sharpe Ratio", f"{sharpe:.2f}")

# Trade log
log_df = df[['datetime', 'Signal', 'Strategy_Return', 'ATR', 'Confidence']].copy()
log_df['Direction'] = log_df['Signal'].replace({0: 'Sell', 1: 'Hold', 2: 'Buy'})
log_df['Stop_Loss'] = log_df['ATR'] * 1.5
log_df['Take_Profit'] = log_df['ATR'] * 2.5
log_df['Strategy_Return'] = log_df['Strategy_Return'].round(4)
log_df['Confidence'] = log_df['Confidence'].round(4)
log_df = log_df[['datetime', 'Direction', 'Confidence', 'Stop_Loss', 'Take_Profit', 'Strategy_Return']]

st.subheader("📋 Trade Log")
st.dataframe(log_df.tail(20).reset_index(drop=True), use_container_width=True)

# US Open Sell Strategy with Volatility Filter
if st.sidebar.checkbox("Enable US Open Sell Strategy"):
    import pytz

    # Convert to US Eastern Time
    df['datetime_utc'] = df['datetime'].dt.tz_localize('UTC')
    df['datetime_est'] = df['datetime_utc'].dt.tz_convert('US/Eastern')
    df['date'] = df['datetime_est'].dt.date
    df['time'] = df['datetime_est'].dt.time

    # Strategy parameters
    SL = 100
    TP = 200
    ATR_THRESHOLD = 15

    us_open_trades = []
    for date in df['date'].unique():
        day_df = df[df['date'] == date]
        entry_row = day_df[(day_df['time'] >= pd.to_datetime("10:00:00").time())].head(1)
        if not entry_row.empty:
            atr_value = entry_row['ATR'].values[0]
            if atr_value < ATR_THRESHOLD:
                continue  # Skip low-volatility days

            entry_price = entry_row['open'].values[0]
            sl = entry_price + SL
            tp = entry_price - TP
            trade_df = day_df[day_df['datetime_est'] > entry_row['datetime_est'].values[0]]
            exit_price = None
            for _, row in trade_df.iterrows():
                if row['high'] >= sl:
                    exit_price = sl
                    result = -SL
                    break
                elif row['low'] <= tp:
                    exit_price = tp
                    result = TP
                    break
            if exit_price is None:
                exit_price = trade_df['close'].values[-1]
                result = entry_price - exit_price
            us_open_trades.append({
                'date': date,
                'entry': entry_price,
                'exit': exit_price,
                'pnl': result,
                'atr': atr_value
            })

    # Convert to DataFrame
    strategy_df = pd.DataFrame(us_open_trades)
    strategy_df['cumulative'] = strategy_df['pnl'].cumsum()

    # Display results
    st.subheader("📉 US Open 30-Min Sell Strategy (Volatility Filtered)")
    st.line_chart(strategy_df.set_index('date')['cumulative'])

    win_rate = (strategy_df['pnl'] > 0).mean()
    avg_gain = strategy_df[strategy_df['pnl'] > 0]['pnl'].mean()
    avg_loss = strategy_df[strategy_df['pnl'] < 0]['pnl'].mean()
    sharpe = strategy_df['pnl'].mean() / strategy_df['pnl'].std() * np.sqrt(252)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Win Rate", f"{win_rate:.2%}")
    col2.metric("Avg Gain", f"{avg_gain:.2f}")
    col3.metric("Avg Loss", f"{avg_loss:.2f}")
    col4.metric("Sharpe Ratio", f"{sharpe:.2f}")

    st.subheader("📋 US Open Strategy Trade Log")
    st.dataframe(strategy_df.tail(20).reset_index(drop=True), use_container_width=True)
