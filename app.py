import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import ta

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="GreenQuant | Stock Intelligence",
    page_icon="📊",
    layout="wide"
)

# ======================================================
# FORCE GREEN BACKGROUND (NO WHITE ANYWHERE)
# ======================================================
st.markdown("""
<style>
html, body, #root, .stApp {
    background: linear-gradient(180deg, #064e3b 0%, #022c22 50%, #021f18 100%) !important;
    color: #ecfdf5;
    font-family: Inter, sans-serif;
}

section.main > div {
    background: transparent !important;
}

header {
    background: transparent !important;
}

/* HERO */
.hero {
    background: linear-gradient(135deg, #16a34a, #065f46);
    padding: 32px;
    border-radius: 22px;
    box-shadow: 0 18px 40px rgba(0,0,0,0.4);
}

/* METRIC CARDS */
.metric {
    background: rgba(6,95,70,0.55);
    backdrop-filter: blur(8px);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
}

.metric-title {
    font-size: 13px;
    color: #bbf7d0;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: #34d399;
}

/* INFO BANNER */
.banner {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: #022c22;
    padding: 16px;
    border-radius: 16px;
    text-align: center;
    font-weight: 600;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #022c22, #021f18) !important;
    border-right: 1px solid rgba(255,255,255,0.15);
}

/* TABS */
button[data-baseweb="tab"] {
    color: #bbf7d0;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #34d399;
    border-bottom: 3px solid #34d399;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HERO HEADER
# ======================================================
st.markdown("""
<div class="hero">
    <h1>GreenQuant</h1>
    <p>Real-Time Market Data • Technical Analysis • Decision Support</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "This platform is intended strictly for educational and analytical purposes. "
    "It does not provide financial or investment advice."
)

st.markdown("---")

# ======================================================
# SIDEBAR CONTROLS
# ======================================================
st.sidebar.header("Market Controls")

stocks = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "NVIDIA (NVDA)": "NVDA",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "Google (GOOGL)": "GOOGL"
}

stock_name = st.sidebar.selectbox("Select Equity", list(stocks.keys()))
ticker = stocks[stock_name]

period = st.sidebar.selectbox("Historical Range", ["6mo", "1y", "2y", "5y"])
ma_period = st.sidebar.slider("Moving Average Period", 5, 100, 20, 5)

# ======================================================
# FETCH DATA
# ======================================================
with st.spinner("Loading market data..."):
    df = yf.download(ticker, period=period)

if df.empty:
    st.error("Unable to load market data.")
    st.stop()

# ======================================================
# INDICATORS
# ======================================================
close = df["Close"].squeeze()
df["RSI"] = ta.momentum.RSIIndicator(close).rsi()
df["MACD"] = ta.trend.MACD(close).macd()
df["MA"] = close.rolling(ma_period).mean()
df.dropna(inplace=True)

# ======================================================
# METRICS (SAFE FLOAT CASTING)
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
metric(c2, "Relative Strength Index", f"{last_rsi:.1f}")
metric(c3, "MACD Value", f"{last_macd:.2f}")
metric(c4, "Data Date", last_date)

# ======================================================
# MARKET CONDITION
# ======================================================
if last_rsi < 30:
    condition = "Oversold conditions detected"
elif last_rsi > 70:
    condition = "Overbought conditions detected"
else:
    condition = "Market momentum appears neutral"

st.markdown(f"""
<div class="banner">
    Market Condition: {condition}
</div>
""", unsafe_allow_html=True)

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3 = st.tabs([
    "Price Analysis",
    "Technical Indicators",
    "Signal Explanation"
])

# ======================================================
# TAB 1: PRICE CHART
# ======================================================
with tab1:
    st.subheader("Price Trend with Moving Average")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df.index, df["Close"], label="Closing Price", linewidth=2, color="#34d399")
    ax.plot(df.index, df["MA"], label=f"{ma_period}-Period Moving Average",
            linewidth=2, color="#bbf7d0")

    ax.grid(color="#065f46", alpha=0.6)
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    **What this chart shows:**  
    The chart displays the historical closing price of the selected equity.
    The moving average smooths short-term price fluctuations and helps identify the overall trend direction.
    """)

# ======================================================
# TAB 2: INDICATORS
# ======================================================
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(df.index, df["RSI"], color="#34d399")
        ax.axhline(70, linestyle="--", color="gray")
        ax.axhline(30, linestyle="--", color="gray")
        ax.set_title("Relative Strength Index")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.markdown(
            "RSI measures momentum. Values below 30 indicate oversold conditions, "
            "while values above 70 indicate overbought conditions."
        )

    with col2:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(df.index, df["MACD"], color="#bbf7d0")
        ax.axhline(0, linestyle="--", color="gray")
        ax.set_title("MACD Indicator")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.markdown(
            "MACD reflects trend strength and direction. "
            "Positive values suggest upward momentum, while negative values indicate downward momentum."
        )

# ======================================================
# TAB 3: SIGNAL EXPLANATION
# ======================================================
with tab3:
    st.markdown("""
    <div class="metric">
        <h3>Analytical Signal Summary</h3>
        <p>
        The current assessment is derived from momentum indicators such as RSI and MACD.
        These indicators help identify potential trend continuation or reversal zones.
        </p>
        <p>
        This signal is intended strictly for analytical and educational use.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("GreenQuant | Visual Market Intelligence Platform")
