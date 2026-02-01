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
    <p>Market Data • Technical Analysis • Predictive Insights</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Educational and analytical use only. Not financial advice."
)

st.markdown("---")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("Market Controls")

STOCKS = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL", "Amazon": "AMZN",
    "Tesla": "TSLA", "NVIDIA": "NVDA", "Meta": "META", "Netflix": "NFLX",
    "AMD": "AMD", "Intel": "INTC", "Adobe": "ADBE", "Oracle": "ORCL",
    "Salesforce": "CRM", "Visa": "V", "Mastercard": "MA", "JPMorgan": "JPM",
    "Coca-Cola": "KO", "Pepsi": "PEP", "Walmart": "WMT", "Disney": "DIS"
}

selected_companies = st.sidebar.multiselect(
    "Select Companies (1–3)",
    options=list(STOCKS.keys()),
    default=["Apple"]
)

period = st.sidebar.selectbox("Time Range", ["6mo", "1y", "2y", "5y"])
ma_short = st.sidebar.slider("Short MA", 5, 30, 20)
ma_long = st.sidebar.slider("Long MA", 30, 120, 50)

if len(selected_companies) == 0:
    st.warning("Please select at least one company.")
    st.stop()

# ======================================================
# LOAD & PROCESS DATA FUNCTION
# ======================================================
def load_stock(ticker):
    df = yf.download(ticker, period=period)
    if df.empty:
        return None

    df = df.reset_index()
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.astype(float)

    df["Close"] = close
    df["RSI"] = ta.momentum.RSIIndicator(close).rsi()

    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MA_S"] = close.rolling(ma_short).mean()
    df["MA_L"] = close.rolling(ma_long).mean()

    return df.dropna()

# ======================================================
# LOAD ALL SELECTED STOCKS
# ======================================================
stock_data = {}

for name in selected_companies:
    data = load_stock(STOCKS[name])
    if data is not None:
        stock_data[name] = data

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3 = st.tabs([
    "Price Comparison",
    "Indicators",
    "AI Prediction"
])

# ======================================================
# TAB 1 — PRICE COMPARISON
# ======================================================
with tab1:
    fig = go.Figure()

    for name, df in stock_data.items():
        normalized = df["Close"] / df["Close"].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=normalized,
            name=f"{name} (Normalized)",
            line=dict(width=3)
        ))

    fig.update_layout(
        height=520,
        hovermode="x unified",
        yaxis_title="Normalized Price (Base = 100)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This chart compares relative performance. "
        "All prices are normalized to start at the same value."
    )

# ======================================================
# TAB 2 — INDICATORS
# ======================================================
with tab2:
    for name, df in stock_data.items():
        st.subheader(name)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["RSI"],
            name="RSI"
        ))

        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")

        fig.update_layout(height=300)

        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# TAB 3 — AI PREDICTION (NO ML CRASHES)
# ======================================================
with tab3:
    for name, df in stock_data.items():
        latest = df.iloc[-1]

        score = 0

        # RSI contribution
        if latest["RSI"] < 30:
            score += 2
        elif latest["RSI"] > 70:
            score -= 2

        # Trend contribution
        if latest["MA_S"] > latest["MA_L"]:
            score += 1
        else:
            score -= 1

        # MACD contribution
        if latest["MACD"] > 0:
            score += 1
        else:
            score -= 1

        if score >= 3:
            decision = "BUY"
        elif score <= -3:
            decision = "SELL"
        else:
            decision = "HOLD"

        confidence = min(abs(score) * 25, 100)

        st.markdown(f"""
        <div class="notice">
        <b>{name}</b><br>
        Decision: {decision}<br>
        Confidence: {confidence}%<br>
        RSI: {latest['RSI']:.1f} | MACD: {latest['MACD']:.2f}
        </div>
        """, unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("GreenQuant | Comparison & AI Module")
