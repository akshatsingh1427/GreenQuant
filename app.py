import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import ta
import os
import datetime

# Optional AI model
try:
    from src.model_builder import build_lstm_model
    AI_AVAILABLE = True
except:
    AI_AVAILABLE = False

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="GreenQuant • AI Stock Dashboard",
    page_icon="📈",
    layout="wide"
)

# ======================================================
# GREEN THEME CSS
# ======================================================
st.markdown("""
<style>
body { background-color: #0b1f17; }
h1, h2, h3 { color: #00ff9c; }

.metric {
    background: linear-gradient(145deg, #0f2e22, #071a13);
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 0 18px rgba(0,255,156,0.25);
}

.metric-title { color: #9fffdc; font-size: 14px; }
.metric-value { color: #00ff9c; font-size: 28px; font-weight: 800; }

.card {
    background: #0f2e22;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 0 20px rgba(0,255,156,0.2);
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.title("📈 GreenQuant — Stock Intelligence Dashboard")
st.subheader("Real Market Data • Technical Analysis • Optional AI")
st.caption("⚠️ Educational use only. Not financial advice.")
st.markdown("---")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("⚙️ Controls")

stocks = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "NVIDIA (NVDA)": "NVDA",
    "Meta (META)": "META",
    "Reliance (RELIANCE.NS)": "RELIANCE.NS",
    "TCS (TCS.NS)": "TCS.NS",
    "Infosys (INFY.NS)": "INFY.NS"
}

stock_name = st.sidebar.selectbox("📌 Select Stock", list(stocks.keys()))
ticker = stocks[stock_name]

period = st.sidebar.selectbox("📅 Time Range", ["6mo", "1y", "2y", "5y"])
show_intraday = st.sidebar.checkbox("⏱ Show Intraday (Today)", True)
use_ai = st.sidebar.checkbox("🤖 Use AI Model", True)

st.sidebar.markdown("---")
st.sidebar.caption("GreenQuant v1.0")

# ======================================================
# FETCH DATA
# ======================================================
df = yf.download(ticker, period=period)

if df.empty:
    st.error("❌ Failed to fetch data")
    st.stop()

# ======================================================
# INDICATORS
# ======================================================
close_series = df["Close"].squeeze()

df["RSI"] = ta.momentum.RSIIndicator(close_series).rsi()
macd = ta.trend.MACD(close_series)
df["MACD"] = macd.macd()
df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()
df.dropna(inplace=True)

# ======================================================
# METRICS
# ======================================================
latest_price = float(df["Close"].iloc[-1])
latest_rsi = float(df["RSI"].iloc[-1])
latest_macd = float(df["MACD"].iloc[-1])
latest_date = df.index[-1].date()

c1, c2, c3, c4 = st.columns(4)

def metric(col, title, value):
    with col:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

metric(c1, "Latest Price", f"{latest_price:.2f}")
metric(c2, "RSI", f"{latest_rsi:.2f}")
metric(c3, "MACD", f"{latest_macd:.4f}")
metric(c4, "Last Date", latest_date)

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price Charts",
    "📊 Indicators",
    "🤖 Signal",
    "ℹ️ About"
])

# ======================================================
# TAB 1: PRICE
# ======================================================
with tab1:
    st.markdown("### Price with Moving Averages")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Close"], label="Close", linewidth=2)
    ax.plot(df.index, df["MA20"], label="MA 20", linestyle="--")
    ax.plot(df.index, df["MA50"], label="MA 50", linestyle="--")
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

    if show_intraday:
        intraday = yf.download(ticker, period="1d", interval="5m")
        if not intraday.empty:
            st.markdown("### ⏱ Today’s Intraday Price (5-minute)")
            fig_i, ax_i = plt.subplots(figsize=(12, 4))
            ax_i.plot(intraday.index, intraday["Close"], color="green", linewidth=2)
            ax_i.set_xlabel("Time")
            ax_i.set_ylabel("Price")
            st.pyplot(fig_i)

# ======================================================
# TAB 2: INDICATORS
# ======================================================
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        fig_rsi, ax_rsi = plt.subplots(figsize=(5, 4))
        ax_rsi.plot(df.index, df["RSI"], color="orange")
        ax_rsi.axhline(70, linestyle="--")
        ax_rsi.axhline(30, linestyle="--")
        ax_rsi.set_title("RSI")
        st.pyplot(fig_rsi)

    with col2:
        fig_macd, ax_macd = plt.subplots(figsize=(5, 4))
        ax_macd.plot(df.index, df["MACD"], color="lime")
        ax_macd.axhline(0, linestyle="--")
        ax_macd.set_title("MACD")
        st.pyplot(fig_macd)

# ======================================================
# TAB 3: SIGNAL (AI + FALLBACK)
# ======================================================
with tab3:
    signal = "➖ NEUTRAL"
    confidence = 0

    if use_ai and AI_AVAILABLE and os.path.exists("models/lstm_weights.weights.h5"):
        model = build_lstm_model()
        model.load_weights("models/lstm_weights.weights.h5")

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df["Close"].values.reshape(-1, 1))
        X = scaled[-60:].reshape(1, 60, 1)
        prob = float(model.predict(X)[0][0])

        if prob > 0.52:
            signal = "📈 UP"
        elif prob < 0.48:
            signal = "📉 DOWN"
        else:
            signal = "➖ NEUTRAL"

        confidence = min(abs(prob - 0.5) * 300, 100)

    else:
        if latest_rsi < 30:
            signal = "📈 UP (Oversold)"
            confidence = 40
        elif latest_rsi > 70:
            signal = "📉 DOWN (Overbought)"
            confidence = 40
        else:
            signal = "➖ NEUTRAL"
            confidence = 20

    st.markdown(f"""
    <div class="card">
        <h2>{signal}</h2>
        <p><b>Confidence:</b> {confidence:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# TAB 4: ABOUT
# ======================================================
with tab4:
    st.markdown("""
### About GreenQuant

**GreenQuant** is an educational stock analysis dashboard.

**Features**
- Real-time (delayed) market data
- Intraday + historical charts
- RSI, MACD, Moving Averages
- Optional AI (LSTM) prediction
- Works even without AI model

**Tech Stack**
- Python
- Streamlit
- yFinance
- TensorFlow (optional)
- Matplotlib

⚠️ Not for trading or investment decisions.
""")

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("GreenQuant • Built for learning & demonstration")
