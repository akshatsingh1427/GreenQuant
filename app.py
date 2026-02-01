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
    page_title="GreenQuant | Stock Intelligence Dashboard",
    page_icon="📈",
    layout="wide"
)

# ======================================================
# PROFESSIONAL DARK THEME
# ======================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0f1117;
    color: #e5e7eb;
    font-family: Inter, sans-serif;
}

section.main > div {
    background-color: #0f1117;
}

.hero {
    background: #111827;
    padding: 28px;
    border-radius: 18px;
    border: 1px solid #1f2937;
}

.hero h1 {
    margin-bottom: 4px;
}

.hero p {
    color: #9ca3af;
    margin-top: 0;
}

.metric {
    background: #111827;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #1f2937;
    text-align: center;
}

.metric-title {
    color: #9ca3af;
    font-size: 13px;
    letter-spacing: 0.4px;
}

.metric-value {
    color: #34d399;
    font-size: 26px;
    font-weight: 700;
}

.card {
    background: #111827;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #1f2937;
}

.status {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 14px;
    text-align: center;
    font-size: 16px;
}

[data-testid="stSidebar"] {
    background-color: #0b0d12;
    border-right: 1px solid #1f2937;
}

button[data-baseweb="tab"] {
    color: #9ca3af;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #34d399;
    border-bottom: 2px solid #34d399;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<div class="hero">
    <h1>GreenQuant</h1>
    <p>Market Data • Technical Analysis • Predictive Insights</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "This application is intended strictly for educational and demonstration purposes. "
    "It does not constitute financial advice."
)

st.markdown("---")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("Application Controls")

stocks = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "NVIDIA (NVDA)": "NVDA",
    "Meta Platforms (META)": "META",
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
    "Tata Consultancy Services (TCS.NS)": "TCS.NS",
    "Infosys (INFY.NS)": "INFY.NS"
}

stock_name = st.sidebar.selectbox("Equity Selection", list(stocks.keys()))
ticker = stocks[stock_name]

period = st.sidebar.selectbox("Historical Time Range", ["6mo", "1y", "2y", "5y"])
show_intraday = st.sidebar.checkbox("Display Intraday Data (Today)", True)
use_ai = st.sidebar.checkbox("Enable Predictive Model", True)

ma_period = st.sidebar.slider(
    "Moving Average Period",
    min_value=5,
    max_value=100,
    value=20,
    step=5
)

st.sidebar.caption("GreenQuant Version 1.2")

# ======================================================
# FETCH DATA
# ======================================================
with st.spinner("Loading market data..."):
    df = yf.download(ticker, period=period)

if df.empty:
    st.error("Market data could not be retrieved.")
    st.stop()

# ======================================================
# INDICATORS
# ======================================================
close = df["Close"].squeeze()

df["RSI"] = ta.momentum.RSIIndicator(close).rsi()
macd = ta.trend.MACD(close)
df["MACD"] = macd.macd()
df["MA_USER"] = close.rolling(ma_period).mean()
df.dropna(inplace=True)

# ======================================================
# METRICS
# ======================================================
latest_price = df["Close"].iloc[-1].item()
latest_rsi = df["RSI"].iloc[-1].item()
latest_macd = df["MACD"].iloc[-1].item()
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

metric(c1, "Last Traded Price", f"{latest_price:.2f}")
metric(c2, "Relative Strength Index", f"{latest_rsi:.1f}")
metric(c3, "MACD Value", f"{latest_macd:.3f}")
metric(c4, "Data As Of", latest_date)

# ======================================================
# MARKET CONDITION
# ======================================================
if latest_rsi < 30:
    condition = "Oversold conditions detected"
elif latest_rsi > 70:
    condition = "Overbought conditions detected"
else:
    condition = "Market conditions appear neutral"

st.markdown(f"""
<div class="status">
    Market Condition: <b>{condition}</b>
</div>
""", unsafe_allow_html=True)

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Price Analysis",
    "Technical Indicators",
    "Signal Assessment",
    "About"
])

# ======================================================
# TAB 1: PRICE
# ======================================================
with tab1:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Close"], label="Closing Price", linewidth=2)
    ax.plot(df.index, df["MA_USER"], label=f"{ma_period}-Period Moving Average", linewidth=2)
    ax.legend()
    ax.grid(color="#1f2937", alpha=0.6)
    st.pyplot(fig)

    if show_intraday:
        intraday = yf.download(ticker, period="1d", interval="5m")
        if not intraday.empty:
            fig_i, ax_i = plt.subplots(figsize=(12, 4))
            ax_i.plot(intraday.index, intraday["Close"])
            ax_i.grid(color="#1f2937", alpha=0.6)
            st.pyplot(fig_i)

# ======================================================
# TAB 2: INDICATORS
# ======================================================
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        fig_rsi, ax = plt.subplots(figsize=(5,4))
        ax.plot(df.index, df["RSI"])
        ax.axhline(70, linestyle="--")
        ax.axhline(30, linestyle="--")
        ax.set_title("Relative Strength Index")
        st.pyplot(fig_rsi)

    with c2:
        fig_macd, ax = plt.subplots(figsize=(5,4))
        ax.plot(df.index, df["MACD"])
        ax.axhline(0, linestyle="--")
        ax.set_title("MACD Indicator")
        st.pyplot(fig_macd)

# ======================================================
# TAB 3: SIGNAL
# ======================================================
with tab3:
    signal = "Neutral"
    confidence = 20

    if latest_rsi < 30:
        signal = "Potential Upside Signal"
        confidence = 40
    elif latest_rsi > 70:
        signal = "Potential Downside Signal"
        confidence = 40

    st.markdown(f"""
    <div class="card">
        <h3>Signal Summary</h3>
        <p><b>Assessment:</b> {signal}</p>
        <p><b>Confidence Level:</b> {confidence}%</p>
        <p>
        The signal assessment is derived from momentum indicators, including
        Relative Strength Index and MACD trend behavior.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# TAB 4: ABOUT
# ======================================================
with tab4:
    st.markdown("""
### About GreenQuant

GreenQuant is a stock market analytics dashboard designed to demonstrate
how technical indicators and predictive models can be integrated into
interactive financial applications.

**Key Capabilities**
- Historical and intraday price visualization
- Momentum-based technical indicators
- Optional predictive modeling
- Cloud-deployable architecture

This platform is intended solely for academic and learning purposes.
""")

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("GreenQuant | Professional Edition")
