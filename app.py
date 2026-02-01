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
    page_icon="📊",
    layout="wide"
)

# ======================================================
# LIGHT GREEN THEME
# ======================================================
st.markdown("""
<style>
html, body, .stApp {
    background: linear-gradient(180deg, #ecfdf5, #d1fae5);
    color: #064e3b;
    font-family: "Segoe UI", sans-serif;
}

.hero {
    background: linear-gradient(135deg, #6ee7b7, #34d399);
    padding: 30px;
    border-radius: 22px;
    box-shadow: 0 12px 28px rgba(0,0,0,0.12);
}

.metric {
    background: white;
    border-radius: 18px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}

.metric-title {
    font-size: 13px;
    color: #065f46;
}

.metric-value {
    font-size: 26px;
    font-weight: 700;
    color: #15803d;
}

.banner {
    background: linear-gradient(135deg, #fde047, #facc15);
    padding: 14px;
    border-radius: 14px;
    text-align: center;
    font-weight: 600;
    color: #422006;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ecfdf5, #d1fae5);
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<div class="hero">
    <h1>GreenQuant</h1>
    <p>Real-Time Market Data • Technical Analysis • Decision Support</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "This platform is designed strictly for educational and analytical purposes. "
    "It does not constitute financial advice."
)

st.markdown("---")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("Market Controls")

stocks = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL", "Amazon": "AMZN",
    "Tesla": "TSLA", "NVIDIA": "NVDA", "Meta": "META", "Netflix": "NFLX",
    "AMD": "AMD", "Intel": "INTC", "IBM": "IBM", "Oracle": "ORCL",
    "Adobe": "ADBE", "Salesforce": "CRM", "PayPal": "PYPL",
    "Uber": "UBER", "Airbnb": "ABNB", "Spotify": "SPOT",
    "Qualcomm": "QCOM", "Cisco": "CSCO", "Broadcom": "AVGO",
    "JPMorgan": "JPM", "Visa": "V", "Mastercard": "MA"
}

company = st.sidebar.selectbox("Select Company", list(stocks.keys()))
ticker = stocks[company]

period = st.sidebar.selectbox("Time Range", ["6mo", "1y", "2y", "5y"])
ma_period = st.sidebar.slider("Moving Average Period", 10, 100, 20, 5)

# ======================================================
# FETCH DATA
# ======================================================
with st.spinner("Loading market data..."):
    df = yf.download(ticker, period=period, progress=False)

if df.empty or "Close" not in df.columns:
    st.error("Market data could not be loaded.")
    st.stop()

# ======================================================
# DATA SANITIZATION (CRITICAL)
# ======================================================
close = pd.Series(df["Close"].to_numpy().flatten(), index=df.index)

df = df.assign(Close=close)
df["RSI"] = ta.momentum.RSIIndicator(close).rsi()
df["MACD"] = ta.trend.MACD(close).macd()
df["MA"] = close.rolling(ma_period).mean()

df = df.dropna()

# ======================================================
# METRICS
# ======================================================
last_price = float(df["Close"].iloc[-1])
last_rsi = float(df["RSI"].iloc[-1])
last_macd = float(df["MACD"].iloc[-1])
last_date = df.index[-1].date()

c1, c2, c3, c4 = st.columns(4)

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
metric(c3, "MACD", f"{last_macd:.2f}")
metric(c4, "Data Date", last_date)

# ======================================================
# AI-STYLE DECISION ENGINE
# ======================================================
score = 0
if last_rsi < 30: score += 1
if last_rsi > 70: score -= 1
if last_macd > 0: score += 1
if last_price > df["MA"].iloc[-1]: score += 1

if score >= 2:
    decision = "BUY"
    explanation = "Momentum and trend indicators align positively."
elif score <= -1:
    decision = "SELL"
    explanation = "Momentum indicators indicate downside risk."
else:
    decision = "HOLD"
    explanation = "Indicators are mixed, suggesting neutral conditions."

confidence = min(abs(score) * 25 + 25, 90)

st.markdown(f"""
<div class="banner">
    Recommendation: <strong>{decision}</strong> | Confidence: {confidence:.0f}%  
    <br>{explanation}
</div>
""", unsafe_allow_html=True)

# ======================================================
# INTERACTIVE PRICE GRAPH (FIXED)
# ======================================================
st.subheader("Price Trend with Moving Average")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Close"],
    mode="lines",
    name="Closing Price",
    line=dict(color="#16a34a", width=2),
    hovertemplate="Date: %{x}<br>Price: %{y:.2f}<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["MA"],
    mode="lines",
    name=f"{ma_period}-Day Moving Average",
    line=dict(color="#f97316", width=2),
    hovertemplate="Date: %{x}<br>MA: %{y:.2f}<extra></extra>"
))

fig.update_layout(
    autosize=True,
    height=520,
    hovermode="x unified",
    plot_bgcolor="white",
    paper_bgcolor="white",
    yaxis=dict(
        range=[
            df["Close"].min() * 0.95,
            df["Close"].max() * 1.05
        ]
    ),
    margin=dict(l=40, r=40, t=40, b=40)
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    "This chart displays the historical closing price along with its moving average. "
    "Hover over the graph to inspect exact values and analyze trend behavior."
)

# ======================================================
# INDICATORS
# ======================================================
st.subheader("Technical Indicators")

col1, col2 = st.columns(2)

with col1:
    st.line_chart(df["RSI"])
    st.caption("RSI measures momentum and overbought/oversold conditions.")

with col2:
    st.line_chart(df["MACD"])
    st.caption("MACD reflects trend direction and momentum strength.")

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("GreenQuant | Interactive Market Intelligence Platform")
