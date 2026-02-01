import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ta

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="GreenQuant | Market Intelligence",
    layout="wide"
)

# ======================================================
# THEME
# ======================================================
st.markdown("""
<style>
html, body, .stApp {
    background-color: #e8fdf3;
    color: #064e3b;
    font-family: Arial, sans-serif;
}

.hero {
    background: linear-gradient(135deg, #4ade80, #22c55e);
    padding: 28px;
    border-radius: 20px;
}

.metric {
    background: #ffffff;
    padding: 16px;
    border-radius: 16px;
    text-align: center;
}

.metric-title {
    font-size: 13px;
    color: #065f46;
}

.metric-value {
    font-size: 26px;
    font-weight: bold;
    color: #15803d;
}

.notice {
    background: #fde047;
    padding: 14px;
    border-radius: 14px;
    color: #422006;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<div class="hero">
    <h1>GreenQuant</h1>
    <p>Market Data • Technical Analysis • Decision Support</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "This application is intended strictly for educational and analytical purposes. "
    "It does not provide financial or investment advice."
)

st.markdown("---")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("Market Controls")

stocks = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    "Meta": "META",
    "Netflix": "NFLX",
    "AMD": "AMD",
    "Intel": "INTC",
    "Adobe": "ADBE",
    "Oracle": "ORCL",
    "Salesforce": "CRM",
    "Visa": "V",
    "Mastercard": "MA",
    "JPMorgan": "JPM",
    "Coca-Cola": "KO",
    "Pepsi": "PEP",
    "Walmart": "WMT",
    "Disney": "DIS"
}

company = st.sidebar.selectbox("Select Company", list(stocks.keys()))
ticker = stocks[company]

period = st.sidebar.selectbox("Time Range", ["6mo", "1y", "2y", "5y"])
ma_period = st.sidebar.slider("Moving Average Period", 10, 100, 20, 5)

# ======================================================
# LOAD DATA (SAFE)
# ======================================================
with st.spinner("Loading market data..."):
    df = yf.download(ticker, period=period, auto_adjust=False)

if df.empty:
    st.error("Unable to load market data.")
    st.stop()

# ======================================================
# FIX DATE
# ======================================================
df = df.reset_index()

if "Date" not in df.columns:
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)

df["Date"] = pd.to_datetime(df["Date"])

# ======================================================
# FIX CLOSE (CRITICAL PART)
# ======================================================
close = df["Close"]

# 🔥 THIS IS THE FIX 🔥
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

close = close.astype(float)

# ======================================================
# INDICATORS (NOW SAFE)
# ======================================================
df["Close_1D"] = close
df["RSI"] = ta.momentum.RSIIndicator(close).rsi()
df["MA"] = close.rolling(ma_period).mean()

df = df.dropna()

# ======================================================
# METRICS
# ======================================================
last_price = df["Close_1D"].iloc[-1]
last_rsi = df["RSI"].iloc[-1]
last_date = df["Date"].iloc[-1].date()

c1, c2, c3 = st.columns(3)

def metric(col, title, value):
    with col:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

metric(c1, "Last Price", f"{last_price:.2f}")
metric(c2, "RSI", f"{last_rsi:.1f}")
metric(c3, "Data Date", last_date)

# ======================================================
# DECISION LOGIC
# ======================================================
if last_rsi < 30:
    decision = "BUY"
    explanation = "The stock appears oversold based on momentum indicators."
elif last_rsi > 70:
    decision = "SELL"
    explanation = "The stock appears overbought based on momentum indicators."
else:
    decision = "HOLD"
    explanation = "Momentum indicators suggest neutral market conditions."

st.markdown(f"""
<div class="notice">
Decision: {decision}<br>
{explanation}
</div>
""", unsafe_allow_html=True)

# ======================================================
# GRAPH (GUARANTEED VISIBLE)
# ======================================================
st.subheader("Price Trend with Moving Average")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Close_1D"],
    name="Closing Price",
    line=dict(color="#16a34a", width=3)
))

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["MA"],
    name=f"{ma_period}-Day Moving Average",
    line=dict(color="#f59e0b", width=3)
))

fig.update_layout(
    height=520,
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    "This chart displays the historical closing price and its moving average. "
    "Hover over the graph to inspect values by date."
)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("GreenQuant | Stable Cloud Build")
