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
# THEME CSS
# ======================================================
st.markdown("""
<style>
body { background-color: #0b1f17; }

.hero {
    background: linear-gradient(135deg, #00ff9c, #00c46a);
    padding: 25px;
    border-radius: 22px;
    color: black;
}

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
# HERO HEADER
# ======================================================
st.markdown("""
<div class="hero">
    <h1>📈 GreenQuant</h1>
    <h4>Explore Markets • Visualize Trends • AI Insights</h4>
</div>
""", unsafe_allow_html=True)

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

ma_period = st.sidebar.slider(
    "📊 Moving Average Period",
    min_value=5,
    max_value=100,
    value=20,
    step=5
)

st.sidebar.caption("GreenQuant v1.1")

# ======================================================
# FETCH DATA
# ======================================================
with st.spinner("Fetching market data..."):
    df = yf.download(ticker, period=period)

if df.empty:
    st.error("❌ Failed to fetch data")
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

metric(c1, "Price", f"{latest_price:.2f}")
metric(c2, "RSI", f"{latest_rsi:.1f}")
metric(c3, "MACD", f"{latest_macd:.3f}")
metric(c4, "Date", latest_date)

# ======================================================
# MARKET MOOD (FUN UX)
# ======================================================
if latest_rsi < 30:
    mood = "😰 Oversold"
    mood_color = "#00ff9c"
elif latest_rsi > 70:
    mood = "😤 Overbought"
    mood_color = "#ff4d4d"
else:
    mood = "😐 Balanced"
    mood_color = "#facc15"

st.markdown(f"""
<div style="padding:15px; border-radius:15px;
background:{mood_color}; color:black; text-align:center;">
    <h3>Market Mood: {mood}</h3>
</div>
""", unsafe_allow_html=True)

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price",
    "📊 Indicators",
    "🤖 Signal",
    "ℹ️ About"
])

# ======================================================
# TAB 1: PRICE
# ======================================================
with tab1:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Close"], label="Close", linewidth=2)
    ax.plot(df.index, df["MA_USER"], label=f"MA {ma_period}", linewidth=2)
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    if show_intraday:
        intraday = yf.download(ticker, period="1d", interval="5m")
        if not intraday.empty:
            fig_i, ax_i = plt.subplots(figsize=(12, 4))
            ax_i.plot(intraday.index, intraday["Close"], color="green")
            ax_i.grid(alpha=0.3)
            st.pyplot(fig_i)

# ======================================================
# TAB 2: INDICATORS
# ======================================================
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        fig_rsi, ax = plt.subplots(figsize=(5,4))
        ax.plot(df.index, df["RSI"], color="orange")
        ax.axhline(70, linestyle="--")
        ax.axhline(30, linestyle="--")
        ax.set_title("RSI")
        st.pyplot(fig_rsi)

    with c2:
        fig_macd, ax = plt.subplots(figsize=(5,4))
        ax.plot(df.index, df["MACD"], color="lime")
        ax.axhline(0, linestyle="--")
        ax.set_title("MACD")
        st.pyplot(fig_macd)

# ======================================================
# TAB 3: SIGNAL
# ======================================================
with tab3:
    signal = "➖ NEUTRAL"
    confidence = 20

    if latest_rsi < 30:
        signal = "📈 UP (Oversold)"
        confidence = 40
    elif latest_rsi > 70:
        signal = "📉 DOWN (Overbought)"
        confidence = 40

    st.markdown(f"""
    <div class="card">
        <h2>{signal}</h2>
        <p><b>Confidence:</b> {confidence}%</p>
        <p>
        RSI: {latest_rsi:.1f}<br>
        MACD Momentum: {"Positive" if latest_macd > 0 else "Negative"}
        </p>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# TAB 4: ABOUT
# ======================================================
with tab4:
    st.markdown("""
### About GreenQuant

GreenQuant is an **interactive stock analysis dashboard** designed for learning.

✨ What makes it fun:
- Live controls (sliders & toggles)
- Market mood indicators
- Interactive charts
- Clean visual design

⚠️ Not for real trading.
""")

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("GreenQuant • Interactive Edition")
