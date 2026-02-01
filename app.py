import streamlit as st
import yfinance as yf
import pandas as pd
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
# LOAD DATA
# ======================================================
with st.spinner("Loading market data..."):
    df = yf.download(ticker, period=period)

if df.empty:
    st.error("Unable to load market data.")
    st.stop()

# ======================================================
# FIX DATE + CLOSE (CRITICAL & SAFE)
# ======================================================
df = df.reset_index()
df.rename(columns={df.columns[0]: "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"])

close = df["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

close = close.astype(float)
df["Close_1D"] = close

# ======================================================
# INDICATORS
# ======================================================
df["RSI"] = ta.momentum.RSIIndicator(df["Close_1D"]).rsi()

macd_indicator = ta.trend.MACD(df["Close_1D"])
df["MACD"] = macd_indicator.macd()
df["MACD_SIGNAL"] = macd_indicator.macd_signal()
df["MACD_HIST"] = macd_indicator.macd_diff()

df["MA"] = df["Close_1D"].rolling(ma_period).mean()

df = df.dropna()

# ======================================================
# METRICS
# ======================================================
last_price = df["Close_1D"].iloc[-1]
last_rsi = df["RSI"].iloc[-1]
last_macd = df["MACD"].iloc[-1]
last_date = df["Date"].iloc[-1].date()

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
metric(c3, "MACD", f"{last_macd:.3f}")
metric(c4, "Data Date", last_date)

# ======================================================
# DECISION LOGIC (RSI + MACD)
# ======================================================
if last_rsi < 30 and last_macd > 0:
    decision = "BUY"
    explanation = "Oversold momentum with improving trend strength."
elif last_rsi > 70 and last_macd < 0:
    decision = "SELL"
    explanation = "Overbought conditions with weakening momentum."
else:
    decision = "HOLD"
    explanation = "Indicators suggest neutral or mixed conditions."

st.markdown(f"""
<div class="notice">
Decision: {decision}<br>
{explanation}
</div>
""", unsafe_allow_html=True)

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3 = st.tabs([
    "Price Trend",
    "RSI Indicator",
    "MACD Indicator"
])

# ======================================================
# PRICE GRAPH
# ======================================================
with tab1:
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
        height=500,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This chart shows the historical closing price and its moving average. "
        "It helps identify trend direction and potential support or resistance zones."
    )

# ======================================================
# RSI GRAPH
# ======================================================
with tab2:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["RSI"],
        name="RSI",
        line=dict(color="#2563eb", width=3)
    ))

    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")

    fig.update_layout(
        height=400,
        yaxis_title="RSI Value"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The Relative Strength Index (RSI) measures momentum. "
        "Values above 70 indicate overbought conditions, while values below 30 indicate oversold conditions."
    )

# ======================================================
# MACD GRAPH
# ======================================================
with tab3:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["MACD"],
        name="MACD",
        line=dict(color="#16a34a", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["MACD_SIGNAL"],
        name="Signal Line",
        line=dict(color="#dc2626", width=2)
    ))

    fig.add_trace(go.Bar(
        x=df["Date"],
        y=df["MACD_HIST"],
        name="Histogram",
        marker_color="#9ca3af"
    ))

    fig.update_layout(
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The MACD indicator shows trend strength and momentum. "
        "Crossovers and histogram changes can signal potential trend reversals."
    )

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("GreenQuant | Technical Indicator Suite")
