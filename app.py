import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
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
# LIGHT GREEN THEME (STREAMLIT CLOUD SAFE)
# ======================================================
st.markdown("""
<style>
html, body, .stApp {
    background: linear-gradient(180deg, #ecfdf5 0%, #d1fae5 100%) !important;
    color: #064e3b;
    font-family: Inter, sans-serif;
}

section.main > div {
    background: transparent !important;
}

/* HERO */
.hero {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    padding: 30px;
    border-radius: 22px;
    color: #022c22;
    box-shadow: 0 20px 35px rgba(0,0,0,0.15);
}

/* METRIC CARDS */
.metric {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(8px);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 12px 25px rgba(0,0,0,0.08);
}

.metric-title {
    font-size: 13px;
    color: #065f46;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #16a34a;
}

/* INFO BANNER */
.banner {
    background: linear-gradient(135deg, #facc15, #fde047);
    padding: 14px;
    border-radius: 14px;
    color: #422006;
    text-align: center;
    font-weight: 600;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ecfdf5, #d1fae5) !important;
    border-right: 1px solid rgba(0,0,0,0.1);
}

/* TABS */
button[data-baseweb="tab"] {
    color: #065f46;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #16a34a;
    border-bottom: 3px solid #16a34a;
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
    "This platform is designed strictly for educational and analytical purposes. "
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
    "Google (GOOGL)": "GOOGL",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
    "Intel (INTC)": "INTC",
    "AMD (AMD)": "AMD"
}

stock_name = st.sidebar.selectbox("Select Company", list(stocks.keys()))
ticker = stocks[stock_name]

period = st.sidebar.selectbox("Historical Time Range", ["6mo", "1y", "2y", "5y"])
ma_period = st.sidebar.slider("Moving Average Period", 5, 100, 20, 5)

# ======================================================
# FETCH DATA
# ======================================================
with st.spinner("Loading market data..."):
    df = yf.download(ticker, period=period)

if df.empty:
    st.error("Market data could not be loaded.")
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

metric(c1, "Last Traded Price", f"{last_price:.2f}")
metric(c2, "Relative Strength Index", f"{last_rsi:.1f}")
metric(c3, "MACD Value", f"{last_macd:.2f}")
metric(c4, "Data As Of", last_date)

# ======================================================
# MARKET CONDITION
# ======================================================
if last_rsi < 30:
    condition = "The stock appears oversold based on momentum indicators."
elif last_rsi > 70:
    condition = "The stock appears overbought based on momentum indicators."
else:
    condition = "Market momentum currently appears neutral."

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
    "Signal Interpretation"
])

# ======================================================
# TAB 1: PRICE
# ======================================================
with tab1:
    st.subheader("Price Trend with Moving Average")

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df.index, df["Close"], label="Closing Price", linewidth=2, color="#16a34a")
    ax.plot(df.index, df["MA"], label=f"{ma_period}-Period Moving Average",
            linewidth=2, color="#fb7185")

    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.info(
        "This chart shows the historical closing price of the selected company. "
        "The moving average smooths short-term fluctuations to highlight the overall trend direction."
    )

# ======================================================
# TAB 2: INDICATORS
# ======================================================
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(df.index, df["RSI"], color="#facc15")
        ax.axhline(70, linestyle="--", color="gray")
        ax.axhline(30, linestyle="--", color="gray")
        ax.set_title("Relative Strength Index (RSI)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.info(
            "RSI measures momentum. Values below 30 may indicate oversold conditions, "
            "while values above 70 may indicate overbought conditions."
        )

    with col2:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(df.index, df["MACD"], color="#16a34a")
        ax.axhline(0, linestyle="--", color="gray")
        ax.set_title("MACD Indicator")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.info(
            "MACD reflects trend direction and strength. "
            "Positive values suggest bullish momentum, while negative values suggest bearish momentum."
        )

# ======================================================
# TAB 3: SIGNAL
# ======================================================
with tab3:
    st.markdown("""
    <div class="metric">
        <h3>Signal Interpretation</h3>
        <p>
        The current signal is derived from momentum-based technical indicators such as RSI and MACD.
        These indicators are commonly used to assess trend strength and potential reversal zones.
        </p>
        <p>
        This information is intended solely for analytical understanding and learning purposes.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("GreenQuant | Interactive Market Intelligence Platform")
