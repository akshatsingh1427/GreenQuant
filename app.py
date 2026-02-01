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
# LIGHT GREEN THEME (CLOUD SAFE)
# ======================================================
st.markdown("""
<style>
html, body, .stApp {
    background: linear-gradient(180deg, #ecfdf5 0%, #d1fae5 100%);
    color: #064e3b;
    font-family: Arial, sans-serif;
}

section.main > div {
    background: transparent !important;
}

.hero {
    background: linear-gradient(135deg, #a7f3d0, #6ee7b7);
    padding: 32px;
    border-radius: 22px;
    box-shadow: 0 18px 35px rgba(0,0,0,0.12);
}

.metric {
    background: white;
    border-radius: 18px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 10px 22px rgba(0,0,0,0.08);
}

.metric-title {
    font-size: 13px;
    color: #065f46;
}

.metric-value {
    font-size: 26px;
    font-weight: 700;
    color: #16a34a;
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
# HERO
# ======================================================
st.markdown("""
<div class="hero">
    <h1>GreenQuant</h1>
    <p>Market Data • Technical Analysis • Intelligent Decision Support</p>
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
    df = yf.download(ticker, period=period)

if df.empty or "Close" not in df:
    st.error("Market data could not be loaded.")
    st.stop()

# ======================================================
# INDICATORS
# ======================================================
close = df["Close"].astype(float)

df["RSI"] = ta.momentum.RSIIndicator(close).rsi()
df["MACD"] = ta.trend.MACD(close).macd()
df["MA"] = close.rolling(ma_period).mean()
df["Returns"] = close.pct_change()
df["Volatility"] = df["Returns"].rolling(20).std()

df.dropna(inplace=True)

# ======================================================
# METRICS
# ======================================================
last_price = float(df["Close"].iloc[-1])
last_rsi = float(df["RSI"].iloc[-1])
last_macd = float(df["MACD"].iloc[-1])
last_vol = float(df["Volatility"].iloc[-1]) * 100
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
metric(c4, "Volatility (20d)", f"{last_vol:.2f}%")

# ======================================================
# AI BUY / HOLD / SELL (EXPLAINABLE)
# ======================================================
score = 0
if last_rsi < 30: score += 1
if last_rsi > 70: score -= 1
if last_macd > 0: score += 1
if last_price > df["MA"].iloc[-1]: score += 1

if score >= 2:
    decision = "BUY"
    explanation = "Momentum and trend indicators suggest potential upside."
elif score <= -1:
    decision = "SELL"
    explanation = "Momentum indicators show potential downside risk."
else:
    decision = "HOLD"
    explanation = "Signals are mixed, suggesting a neutral market condition."

confidence = min(abs(score) * 25 + 25, 90)

st.markdown(f"""
<div class="banner">
    AI Recommendation: <strong>{decision}</strong> | Confidence: {confidence:.0f}%  
    <br>{explanation}
</div>
""", unsafe_allow_html=True)

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3 = st.tabs([
    "Price Analysis",
    "Technical Indicators",
    "Interpretation"
])

# ======================================================
# TAB 1: INTERACTIVE PRICE GRAPH
# ======================================================
with tab1:
    st.subheader("Price Trend with Moving Average")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        name="Closing Price",
        line=dict(color="#16a34a", width=2),
        hovertemplate="Date: %{x}<br>Price: %{y:.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["MA"],
        name=f"{ma_period}-Day Moving Average",
        line=dict(color="#fb7185", width=2),
        hovertemplate="Date: %{x}<br>MA: %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        height=520,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This interactive chart shows the historical closing price and its moving average. "
        "You can zoom, pan, and hover to inspect price behavior at specific dates."
    )

# ======================================================
# TAB 2: INDICATORS
# ======================================================
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Relative Strength Index (RSI)")
        st.line_chart(df["RSI"])
        st.caption(
            "RSI measures momentum. Values below 30 may indicate oversold conditions, "
            "while values above 70 may indicate overbought conditions."
        )

    with col2:
        st.subheader("MACD Indicator")
        st.line_chart(df["MACD"])
        st.caption(
            "MACD represents trend direction and momentum. "
            "Positive values suggest bullish momentum; negative values suggest bearish momentum."
        )

# ======================================================
# TAB 3: INTERPRETATION
# ======================================================
with tab3:
    st.markdown("""
    ### How to Read These Signals

    - **Price vs Moving Average** indicates trend direction.
    - **RSI** highlights momentum extremes.
    - **MACD** reflects trend strength and reversals.
    - **Volatility** measures recent price instability.

    The AI recommendation combines these indicators into a simple, explainable decision.
    """)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("GreenQuant | Interactive Market Intelligence Platform")
