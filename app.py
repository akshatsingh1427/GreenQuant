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
    color: #052e16;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
}

.metric {
    background: #ffffff;
    padding: 16px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 6px 16px rgba(0,0,0,0.1);
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
    background: #fef08a;
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
    "This platform is for educational and analytical purposes only. "
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
    "IBM": "IBM",
    "Adobe": "ADBE",
    "Oracle": "ORCL",
    "Salesforce": "CRM",
    "PayPal": "PYPL",
    "Uber": "UBER",
    "Airbnb": "ABNB",
    "Visa": "V",
    "Mastercard": "MA",
    "JPMorgan": "JPM"
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

if df.empty or "Close" not in df:
    st.error("Unable to load market data.")
    st.stop()

# ======================================================
# DATA CLEANING (CRITICAL)
# ======================================================
df = df.reset_index()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])

df["Close"] = df["Close"].astype(float)
df = df.sort_values("Date")

df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
df["MA"] = df["Close"].rolling(ma_period).mean()

df = df.dropna()

# ======================================================
# METRICS
# ======================================================
last_price = df["Close"].iloc[-1]
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
# DECISION ENGINE
# ======================================================
if last_rsi < 30:
    decision = "BUY"
    explanation = "The stock appears oversold based on RSI."
elif last_rsi > 70:
    decision = "SELL"
    explanation = "The stock appears overbought based on RSI."
else:
    decision = "HOLD"
    explanation = "Momentum indicators suggest neutral conditions."

st.markdown(f"""
<div class="notice">
Decision: {decision}<br>
{explanation}
</div>
""", unsafe_allow_html=True)

# ======================================================
# GRAPH (FIXED & VISIBLE)
# ======================================================
st.subheader("Price Trend with Moving Average")

y_min = df["Close"].min() * 0.97
y_max = df["Close"].max() * 1.03

fig = go.Figure()

fig.add_trace(go.Scattergl(
    x=df["Date"],
    y=df["Close"],
    name="Closing Price",
    line=dict(color="#16a34a", width=3)
))

fig.add_trace(go.Scattergl(
    x=df["Date"],
    y=df["MA"],
    name=f"{ma_period}-Day Moving Average",
    line=dict(color="#f97316", width=3)
))

fig.update_layout(
    height=520,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=50, r=50, t=40, b=40),
    yaxis=dict(range=[y_min, y_max]),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    "This chart displays the historical closing price along with a moving average. "
    "Use the hover tool to inspect prices on specific dates."
)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("GreenQuant | Stable Release")
