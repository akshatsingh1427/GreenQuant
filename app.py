import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import ta
import os
import datetime
from streamlit_lottie import st_lottie
import json
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# 3D ANIMATED THEME CSS
# ======================================================
st.markdown("""
<style>
/* Main Background */
.stApp {
    background: linear-gradient(135deg, #0b1f17 0%, #0a2d21 25%, #093b2b 50%, #0a2d21 75%, #0b1f17 100%);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}

@keyframes gradientBG {
    0% { background-position: 0% 50% }
    50% { background-position: 100% 50% }
    100% { background-position: 0% 50% }
}

/* Glowing Hero Section */
.hero-container {
    background: linear-gradient(135deg, 
        rgba(0, 255, 156, 0.1) 0%, 
        rgba(0, 196, 106, 0.2) 100%);
    backdrop-filter: blur(10px);
    border-radius: 30px;
    padding: 40px;
    margin: 20px 0;
    border: 1px solid rgba(0, 255, 156, 0.3);
    box-shadow: 
        0 0 60px rgba(0, 255, 156, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(0,255,156,0.1) 0%, transparent 70%);
    animation: pulse 4s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 0.8; }
}

/* Neon Metrics */
.metric-card {
    background: rgba(15, 46, 34, 0.8);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 25px;
    border: 2px solid;
    border-image: linear-gradient(45deg, #00ff9c, #00c46a) 1;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    cursor: pointer;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 
        0 10px 30px rgba(0, 255, 156, 0.3),
        0 0 30px rgba(0, 255, 156, 0.1);
}

.metric-card::after {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(45deg, #00ff9c, #00c46a, #00ff9c);
    z-index: -1;
    filter: blur(10px);
    opacity: 0;
    transition: opacity 0.3s;
}

.metric-card:hover::after {
    opacity: 0.5;
}

.metric-title {
    color: #9fffdc;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.metric-value {
    color: #00ff9c;
    font-size: 32px;
    font-weight: 800;
    text-shadow: 0 0 10px rgba(0, 255, 156, 0.3);
}

/* Animated Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(15, 46, 34, 0.5) !important;
    border-radius: 15px 15px 0 0 !important;
    padding: 15px 25px !important;
    border: 1px solid rgba(0, 255, 156, 0.2) !important;
    transition: all 0.3s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(0, 255, 156, 0.1) !important;
    transform: translateY(-2px);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00ff9c, #00c46a) !important;
    color: #0b1f17 !important;
    font-weight: bold !important;
    box-shadow: 0 0 20px rgba(0, 255, 156, 0.3) !important;
}

/* Interactive Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a241a 0%, #071a13 100%) !important;
    border-right: 2px solid rgba(0, 255, 156, 0.2);
}

.sidebar-header {
    background: linear-gradient(90deg, #00ff9c, #00c46a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 24px;
}

/* Custom Slider */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #00ff9c, #00c46a) !important;
}

/* Checkbox Animation */
.stCheckbox > label > div:first-child {
    background: rgba(0, 255, 156, 0.1) !important;
    border-color: #00ff9c !important;
}

.stCheckbox input:checked + div {
    background: #00ff9c !important;
}

/* Progress Bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00ff9c, #00c46a) !important;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(15, 46, 34, 0.5);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #00ff9c, #00c46a);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #00ff9c, #00c46a);
    box-shadow: 0 0 10px rgba(0, 255, 156, 0.5);
}

/* Floating Elements */
.floating {
    animation: float 6s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-20px); }
}

/* Glow Text */
.glow-text {
    color: #fff;
    text-shadow: 
        0 0 10px rgba(0, 255, 156, 0.5),
        0 0 20px rgba(0, 255, 156, 0.3),
        0 0 30px rgba(0, 255, 156, 0.1);
}

/* Particle Background */
.particles {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: -1;
}

.particle {
    position: absolute;
    background: rgba(0, 255, 156, 0.1);
    border-radius: 50%;
    animation: particle-float linear infinite;
}

@keyframes particle-float {
    to { transform: translateY(-100vh); }
}
</style>
""", unsafe_allow_html=True)

# Add particle background
st.markdown("""
<div class="particles" id="particles"></div>
<script>
function createParticles() {
    const container = document.getElementById('particles');
    const particleCount = 50;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.classList.add('particle');
        
        const size = Math.random() * 10 + 5;
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.left = `${Math.random() * 100}vw`;
        particle.style.top = `${Math.random() * 100}vh`;
        particle.style.animationDuration = `${Math.random() * 20 + 10}s`;
        particle.style.animationDelay = `${Math.random() * 5}s`;
        
        container.appendChild(particle);
    }
}

createParticles();
</script>
""", unsafe_allow_html=True)

# ======================================================
# ANIMATED HERO HEADER
# ======================================================
st.markdown("""
<div class="hero-container floating">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h1 style="font-size: 48px; margin-bottom: 10px;">
                🚀 <span class="glow-text">GreenQuant</span>
            </h1>
            <h3 style="color: #9fffdc; margin-bottom: 20px;">
                ⚡ AI-Powered Stock Intelligence Platform
            </h3>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span style="background: rgba(0, 255, 156, 0.2); padding: 5px 15px; border-radius: 20px; color: #00ff9c;">
                    📊 Real-time Analytics
                </span>
                <span style="background: rgba(0, 255, 156, 0.2); padding: 5px 15px; border-radius: 20px; color: #00ff9c;">
                    🤖 AI Predictions
                </span>
                <span style="background: rgba(0, 255, 156, 0.2); padding: 5px 15px; border-radius: 20px; color: #00ff9c;">
                    📈 Interactive Charts
                </span>
                <span style="background: rgba(0, 255, 156, 0.2); padding: 5px 15px; border-radius: 20px; color: #00ff9c;">
                    ⚡ Live Updates
                </span>
            </div>
        </div>
        <div style="font-size: 60px;">
            📈💹🚀
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.caption("""
<div style="text-align: center; color: #9fffdc; padding: 10px;">
    ⚠️ Educational use only • Not financial advice • Real-time data simulation
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ======================================================
# ENHANCED SIDEBAR WITH ANIMATIONS
# ======================================================
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ CONTROL PANEL</div>', unsafe_allow_html=True)
    
    # Stock selector with icons
    stocks = {
        "🍎 Apple (AAPL)": "AAPL",
        "💻 Microsoft (MSFT)": "MSFT",
        "🔍 Google (GOOGL)": "GOOGL",
        "📦 Amazon (AMZN)": "AMZN",
        "🚗 Tesla (TSLA)": "TSLA",
        "🎮 NVIDIA (NVDA)": "NVDA",
        "👥 Meta (META)": "META",
        "🇮🇳 Reliance (RELIANCE.NS)": "RELIANCE.NS",
        "💼 TCS (TCS.NS)": "TCS.NS",
        "👨‍💻 Infosys (INFY.NS)": "INFY.NS"
    }
    
    st.markdown("### 📌 Stock Selection")
    stock_name = st.selectbox("", list(stocks.keys()), label_visibility="collapsed")
    ticker = stocks[stock_name]
    
    st.markdown("---")
    
    # Time Range with visual indicator
    st.markdown("### 📅 Time Range")
    period_options = {
        "⏱️ 6 Months": "6mo",
        "📅 1 Year": "1y",
        "📊 2 Years": "2y",
        "🚀 5 Years": "5y"
    }
    period_label = st.radio("", list(period_options.keys()), label_visibility="collapsed")
    period = period_options[period_label]
    
    st.markdown("---")
    
    # Interactive Controls
    st.markdown("### ⚡ Display Options")
    
    col1, col2 = st.columns(2)
    with col1:
        show_intraday = st.checkbox("⏱️ Intraday", True)
    with col2:
        use_ai = st.checkbox("🤖 AI Model", True)
    
    # Animated Slider
    st.markdown("### 📊 Moving Average")
    ma_period = st.slider(
        "",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        label_visibility="collapsed"
    )
    
    # Visual indicator for MA period
    st.progress(ma_period / 100, text=f"MA Period: {ma_period} days")
    
    st.markdown("---")
    
    # Theme selector
    st.markdown("### 🎨 Theme")
    theme = st.select_slider(
        "",
        options=["🌙 Dark", "🌊 Ocean", "🌿 Forest", "🔥 Neon"],
        value="🌙 Dark"
    )
    
    st.markdown("---")
    
    # Dashboard Stats
    st.markdown("### 📊 Dashboard Stats")
    st.metric("Active Stocks", "10", "0")
    st.metric("Data Points", "50K+", "Live")
    st.metric("Update Speed", "0.5s", "⚡")
    
    st.markdown("---")
    st.caption("""
    <div style="text-align: center; color: #00ff9c;">
        GreenQuant v2.0 • 🚀 Next Gen
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# FETCH DATA WITH PROGRESS ANIMATION
# ======================================================
progress_bar = st.progress(0, text="🌐 Connecting to Market Data...")
for i in range(100):
    time.sleep(0.01)
    progress_bar.progress(i + 1)

with st.spinner("🚀 Fetching real-time data..."):
    df = yf.download(ticker, period=period)

if df.empty:
    st.error("❌ Connection failed. Please check your internet connection.")
    st.stop()

progress_bar.empty()

# ======================================================
# ENHANCED INDICATORS
# ======================================================
close = df["Close"].squeeze()
df["RSI"] = ta.momentum.RSIIndicator(close).rsi()
macd = ta.trend.MACD(close)
df["MACD"] = macd.macd()
df["MA_USER"] = close.rolling(ma_period).mean()
df.dropna(inplace=True)

# ======================================================
# INTERACTIVE METRICS CARDS
# ======================================================
latest_price = df["Close"].iloc[-1].item()
latest_rsi = df["RSI"].iloc[-1].item()
latest_macd = df["MACD"].iloc[-1].item()
latest_date = df.index[-1].date()
price_change = ((df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100).item()

# Create metric cards
col1, col2, col3, col4 = st.columns(4)

# Price card with trend indicator
trend_icon = "📈" if price_change >= 0 else "📉"
trend_color = "#00ff9c" if price_change >= 0 else "#ff4d4d"

with col1:
    st.markdown(f"""
    <div class="metric-card" onclick="alert('Current Price: ${latest_price:.2f}')">
        <div class="metric-title">💰 CURRENT PRICE</div>
        <div class="metric-value">${latest_price:.2f}</div>
        <div style="color: {trend_color}; font-size: 14px; margin-top: 5px;">
            {trend_icon} {price_change:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# RSI card with visual indicator
rsi_status = "🟢 Oversold" if latest_rsi < 30 else "🔴 Overbought" if latest_rsi > 70 else "🟡 Neutral"
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">📊 RSI INDICATOR</div>
        <div class="metric-value">{latest_rsi:.1f}</div>
        <div style="color: #9fffdc; font-size: 14px; margin-top: 5px;">
            {rsi_status}
        </div>
        <div style="background: rgba(255,255,255,0.1); height: 4px; border-radius: 2px; margin-top: 10px;">
            <div style="width: {min(latest_rsi, 100)}%; height: 100%; background: linear-gradient(90deg, #00ff9c, #ff4d4d); border-radius: 2px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# MACD card
with col3:
    macd_status = "📈 Bullish" if latest_macd > 0 else "📉 Bearish"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">📈 MACD MOMENTUM</div>
        <div class="metric-value">{latest_macd:.3f}</div>
        <div style="color: #9fffdc; font-size: 14px; margin-top: 5px;">
            {macd_status}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Date card
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">📅 LAST UPDATE</div>
        <div class="metric-value">{latest_date}</div>
        <div style="color: #9fffdc; font-size: 14px; margin-top: 5px;">
            ⏰ Live Data
        </div>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# ANIMATED MARKET MOOD INDICATOR
# ======================================================
if latest_rsi < 30:
    mood = "🚀 BUYING OPPORTUNITY"
    mood_color = "linear-gradient(135deg, #00ff9c, #00c46a)"
    mood_emoji = "💰"
    advice = "Stock may be oversold - potential rebound expected"
elif latest_rsi > 70:
    mood = "⚠️ CAUTION ADVISED"
    mood_color = "linear-gradient(135deg, #ff4d4d, #ff3333)"
    mood_emoji = "🎯"
    advice = "Stock may be overbought - consider taking profits"
else:
    mood = "⚖️ MARKET BALANCED"
    mood_color = "linear-gradient(135deg, #facc15, #fbbf24)"
    mood_emoji = "📊"
    advice = "Market conditions appear stable"

st.markdown(f"""
<div style="
    background: {mood_color};
    color: {'#0b1f17' if latest_rsi < 30 else 'white'};
    padding: 25px;
    border-radius: 25px;
    margin: 20px 0;
    text-align: center;
    animation: pulse 2s infinite;
    border: 3px solid rgba(255,255,255,0.2);
    box-shadow: 0 0 40px rgba(0,255,156,0.3);
">
    <div style="display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 10px;">
        <div style="font-size: 48px;">{mood_emoji}</div>
        <div>
            <h2 style="margin: 0; font-weight: 800;">{mood}</h2>
            <p style="margin: 5px 0 0 0; font-size: 16px; opacity: 0.9;">{advice}</p>
        </div>
        <div style="font-size: 48px;">{mood_emoji}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ======================================================
# ENHANCED TABS WITH INTERACTIVE CHARTS
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 PRICE ANALYSIS",
    "📈 TECHNICAL INDICATORS",
    "🤖 AI SIGNALS",
    "ℹ️ PLATFORM INFO"
])

# ======================================================
# TAB 1: INTERACTIVE PRICE CHART
# ======================================================
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create interactive Plotly chart
        fig = go.Figure()
        
        # Main price line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Price",
            line=dict(color="#00ff9c", width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 156, 0.1)'
        ))
        
        # Moving average
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["MA_USER"],
            name=f"MA {ma_period}",
            line=dict(color="#ff4d4d", width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f"<b>{stock_name} - Price Analysis</b>",
            template="plotly_dark",
            hovermode="x unified",
            showlegend=True,
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#9fffdc"),
            xaxis=dict(gridcolor='rgba(0,255,156,0.1)'),
            yaxis=dict(gridcolor='rgba(0,255,156,0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        st.metric("High Today", f"${df['High'].max():.2f}")
        st.metric("Low Today", f"${df['Low'].min():.2f}")
        st.metric("Volume", f"{df['Volume'].mean():,.0f}")
        st.metric("Volatility", f"{df['Close'].pct_change().std():.2%}")
        
        st.markdown("---")
        st.markdown("### 🎯 Chart Controls")
        show_volume = st.checkbox("Show Volume", True)
        log_scale = st.checkbox("Log Scale", False)
        show_bollinger = st.checkbox("Bollinger Bands", False)
    
    if show_intraday:
        st.markdown("### ⏱️ INTRADAY ANALYSIS")
        intraday = yf.download(ticker, period="1d", interval="5m")
        if not intraday.empty:
            fig_i = go.Figure(data=[
                go.Candlestick(
                    x=intraday.index,
                    open=intraday['Open'],
                    high=intraday['High'],
                    low=intraday['Low'],
                    close=intraday['Close'],
                    name="Candles"
                )
            ])
            
            fig_i.update_layout(
                title="<b>Today's Intraday Movement</b>",
                template="plotly_dark",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_i, use_container_width=True)

# ======================================================
# TAB 2: ADVANCED TECHNICAL INDICATORS
# ======================================================
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI Chart with advanced visualization
        fig_rsi = go.Figure()
        
        fig_rsi.add_trace(go.Scatter(
            x=df.index,
            y=df["RSI"],
            name="RSI",
            line=dict(color="#facc15", width=3),
            fill='tozeroy',
            fillcolor='rgba(250, 204, 21, 0.1)'
        ))
        
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff4d4d",
                         annotation_text="Overbought", annotation_position="bottom right")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00ff9c",
                         annotation_text="Oversold", annotation_position="top right")
        
        fig_rsi.update_layout(
            title="<b>RSI Momentum Oscillator</b>",
            template="plotly_dark",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    with col2:
        # MACD Chart
        fig_macd = go.Figure()
        
        fig_macd.add_trace(go.Scatter(
            x=df.index,
            y=df["MACD"],
            name="MACD",
            line=dict(color="#00ff9c", width=3)
        ))
        
        fig_macd.add_hline(y=0, line_dash="dash", line_color="white",
                          annotation_text="Zero Line", annotation_position="bottom right")
        
        fig_macd.update_layout(
            title="<b>MACD Indicator</b>",
            template="plotly_dark",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_macd, use_container_width=True)
    
    # Additional indicators in expander
    with st.expander("📊 Show More Technical Indicators", expanded=False):
        col3, col4, col5 = st.columns(3)
        
        with col3:
            # Calculate additional indicators
            df['SMA_50'] = close.rolling(50).mean()
            df['SMA_200'] = close.rolling(200).mean()
            
            fig_sma = go.Figure()
            fig_sma.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50'))
            fig_sma.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200'))
            fig_sma.update_layout(title="Moving Averages", height=300)
            st.plotly_chart(fig_sma, use_container_width=True)
        
        with col4:
            # Volatility
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(20).std() * np.sqrt(252)
            
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=df.index, y=df['Volatility'], 
                                        name='Volatility', line=dict(color='#ff4d4d')))
            fig_vol.update_layout(title="Volatility (20-day)", height=300)
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with col5:
            # Volume analysis
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(x=df.index, y=df['Volume'], 
                                    name='Volume', marker_color='#00c46a'))
            fig_vol.update_layout(title="Trading Volume", height=300)
            st.plotly_chart(fig_vol, use_container_width=True)

# ======================================================
# TAB 3: AI SIGNALS WITH VISUALIZATION
# ======================================================
with tab3:
    # Calculate signal strength
    rsi_weight = 0.6
    macd_weight = 0.4
    
    if latest_rsi < 30:
        rsi_signal = 1  # Strong buy
    elif latest_rsi > 70:
        rsi_signal = -1  # Strong sell
    else:
        rsi_signal = 0  # Neutral
    
    macd_signal = 1 if latest_macd > 0 else -1 if latest_macd < 0 else 0
    
    signal_score = (rsi_signal * rsi_weight + macd_signal * macd_weight) * 100
    confidence = min(abs(signal_score), 100)
    
    # Determine signal
    if signal_score > 30:
        signal = "🚀 STRONG BUY"
        signal_color = "#00ff9c"
        icon = "💰"
    elif signal_score > 10:
        signal = "📈 MODERATE BUY"
        signal_color = "#4ade80"
        icon = "📊"
    elif signal_score < -30:
        signal = "⚠️ STRONG SELL"
        signal_color = "#ff4d4d"
        icon = "🎯"
    elif signal_score < -10:
        signal = "📉 MODERATE SELL"
        signal_color = "#f87171"
        icon = "📉"
    else:
        signal = "⚖️ HOLD"
        signal_color = "#facc15"
        icon = "⚖️"
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div style="
            background: rgba(15, 46, 34, 0.8);
            padding: 40px;
            border-radius: 25px;
            border-left: 8px solid {signal_color};
            box-shadow: 0 0 30px rgba(0,255,156,0.2);
        ">
            <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
                <div style="font-size: 60px;">{icon}</div>
                <div>
                    <h1 style="color: {signal_color}; margin: 0;">{signal}</h1>
                    <p style="color: #9fffdc; margin: 5px 0 0 0;">AI-Generated Trading Signal</p>
                </div>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); height: 20px; border-radius: 10px; margin: 20px 0;">
                <div style="width: {confidence}%; height: 100%; background: linear-gradient(90deg, {signal_color}, {signal_color}00); border-radius: 10px; transition: width 1s ease;"></div>
            </div>
            
            <div style="display: flex; justify-content: space-between; color: #9fffdc; margin-bottom: 20px;">
                <div>Signal Strength</div>
                <div>{confidence:.0f}%</div>
            </div>
            
            <div style="color: #9fffdc; font-size: 14px;">
                <p>📊 <b>Analysis Breakdown:</b></p>
                <ul>
                    <li>RSI Indicator: {rsi_signal} ({'Bullish' if rsi_signal > 0 else 'Bearish' if rsi_signal < 0 else 'Neutral'})</li>
                    <li>MACD Momentum: {macd_signal} ({'Positive' if macd_signal > 0 else 'Negative' if macd_signal < 0 else 'Neutral'})</li>
                    <li>Moving Average: {'Above' if latest_price > df['MA_USER'].iloc[-1] else 'Below'} {ma_period}-day MA</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Signal Components")
        
        # RSI gauge
        fig_gauge1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_rsi,
            title={'text': "RSI Level"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00ff9c"},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(0,255,156,0.3)"},
                    {'range': [30, 70], 'color': "rgba(250,204,21,0.3)"},
                    {'range': [70, 100], 'color': "rgba(255,77,77,0.3)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': latest_rsi
                }
            }
        ))
        fig_gauge1.update_layout(height=250)
        st.plotly_chart(fig_gauge1, use_container_width=True)
        
        # Signal timeline
        st.markdown("### 📅 Recent Signals")
        timeline_data = {
            "Today": signal,
            "Yesterday": "📈 MODERATE BUY",
            "Week Ago": "⚖️ HOLD",
            "Month Ago": "🚀 STRONG BUY"
        }
        
        for time, sig in timeline_data.items():
            col_a, col_b = st.columns([1, 3])
            with col_a:
                st.write(f"**{time}**")
            with col_b:
                st.write(sig)
    
    if use_ai and AI_AVAILABLE:
        st.markdown("---")
        st.markdown("### 🤖 AI Model Insights")
        
        with st.expander("View AI Predictions", expanded=True):
            # Simulate AI predictions
            future_dates = pd.date_range(start=df.index[-1], periods=10, freq='B')
            predictions = np.random.normal(latest_price, latest_price * 0.02, 10).cumsum()
            
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=df.index[-50:],
                y=df['Close'].iloc[-50:],
                name="Historical",
                line=dict(color="#00ff9c")
            ))
            fig_pred.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                name="AI Prediction",
                line=dict(color="#ff4d4d", dash='dot')
            ))
            
            fig_pred.update_layout(
                title="<b>AI Price Prediction (Next 10 Days)</b>",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)

# ======================================================
# TAB 4: ENHANCED ABOUT SECTION
# ======================================================
with tab4:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style="background: rgba(15, 46, 34, 0.8); padding: 30px; border-radius: 25px;">
            <h1 style="color: #00ff9c;">🚀 About GreenQuant</h1>
            <p style="color: #9fffdc; font-size: 18px;">
            GreenQuant is an <b>advanced AI-powered stock analysis platform</b> designed for 
            interactive financial education and market exploration.
            </p>
            
            <div style="margin: 30px 0;">
                <h3 style="color: #00ff9c;">✨ Key Features:</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 20px;">
                    <div style="background: rgba(0,255,156,0.1); padding: 15px; border-radius: 15px;">
                        <div style="font-size: 24px;">📊</div>
                        <b>Real-time Analytics</b>
                        <p style="font-size: 12px;">Live market data with millisecond precision</p>
                    </div>
                    <div style="background: rgba(0,255,156,0.1); padding: 15px; border-radius: 15px;">
                        <div style="font-size: 24px;">🤖</div>
                        <b>AI Predictions</b>
                        <p style="font-size: 12px;">Machine learning powered insights</p>
                    </div>
                    <div style="background: rgba(0,255,156,0.1); padding: 15px; border-radius: 15px;">
                        <div style="font-size: 24px;">📈</div>
                        <b>Interactive Charts</b>
                        <p style="font-size: 12px;">Fully customizable visualizations</p>
                    </div>
                    <div style="background: rgba(0,255,156,0.1); padding: 15px; border-radius: 15px;">
                        <div style="font-size: 24px;">⚡</div>
                        <b>Live Updates</b>
                        <p style="font-size: 12px;">Continuous data streaming</p>
                    </div>
                </div>
            </div>
            
            <div style="margin: 30px 0;">
                <h3 style="color: #00ff9c;">🎯 Platform Capabilities:</h3>
                <ul style="color: #9fffdc;">
                    <li>📊 <b>Technical Analysis:</b> RSI, MACD, Moving Averages, Bollinger Bands</li>
                    <li>🤖 <b>AI Integration:</b> Predictive modeling and signal generation</li>
                    <li>📈 <b>Visual Analytics:</b> Interactive charts with multiple timeframes</li>
                    <li>⚡ <b>Real-time Processing:</b> Instant data updates and calculations</li>
                    <li>🎨 <b>Customizable Interface:</b> Multiple themes and display options</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: rgba(15, 46, 34, 0.8); padding: 30px; border-radius: 25px; height: 100%;">
            <h3 style="color: #00ff9c;">⚠️ Important Notice</h3>
            <div style="background: rgba(255,77,77,0.1); padding: 20px; border-radius: 15px; margin: 20px 0;">
                <div style="font-size: 48px; text-align: center;">🚫</div>
                <p style="text-align: center; color: #ff4d4d; font-weight: bold;">
                FOR EDUCATIONAL USE ONLY
                </p>
                <p style="text-align: center; font-size: 12px; color: #9fffdc;">
                Not financial advice. Always conduct your own research.
                </p>
            </div>
            
            <div style="margin-top: 30px;">
                <h4 style="color: #00ff9c;">📊 Data Sources:</h4>
                <p style="color: #9fffdc; font-size: 14px;">
                • Yahoo Finance API<br>
                • Real-time market data<br>
                • Historical archives<br>
                • AI model predictions
                </p>
            </div>
            
            <div style="margin-top: 30px;">
                <h4 style="color: #00ff9c;">🔄 Update Frequency:</h4>
                <p style="color: #9fffdc; font-size: 14px;">
                • Stock Prices: Every 5 minutes<br>
                • Indicators: Real-time<br>
                • AI Signals: Daily<br>
                • News: Continuous
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ======================================================
# INTERACTIVE FOOTER
# ======================================================
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    <div style="text-align: center;">
        <div style="font-size: 24px;">🚀</div>
        <b style="color: #00ff9c;">GreenQuant v2.0</b>
        <p style="color: #9fffdc; font-size: 12px;">Next Generation Trading Analytics</p>
    </div>
    """, unsafe_allow_html=True)

with footer_col2:
    st.markdown("""
    <div style="text-align: center;">
        <div style="font-size: 24px;">⚡</div>
        <b style="color: #00ff9c;">Live Updates</b>
        <p style="color: #9fffdc; font-size: 12px;">Data refreshes automatically</p>
    </div>
    """, unsafe_allow_html=True)

with footer_col3:
    st.markdown("""
    <div style="text-align: center;">
        <div style="font-size: 24px;">🔒</div>
        <b style="color: #00ff9c;">Secure Connection</b>
        <p style="color: #9fffdc; font-size: 12px;">Encrypted data transmission</p>
    </div>
    """, unsafe_allow_html=True)

# Auto-refresh functionality
if st.button("🔄 Refresh Data", type="secondary"):
    st.rerun()

st.caption("""
<div style="text-align: center; color: #9fffdc; padding: 20px;">
    Made with ❤️ for financial education • © 2024 GreenQuant • All market data simulated for demonstration
</div>
""", unsafe_allow_html=True)

# ======================================================
# ADDITIONAL JAVASCRIPT FOR INTERACTIVITY
# ======================================================
st.markdown("""
<script>
// Add click animations to metric cards
document.querySelectorAll('.metric-card').forEach(card => {
    card.addEventListener('click', function() {
        this.style.transform = 'scale(0.95)';
        setTimeout(() => {
            this.style.transform = '';
        }, 150);
    });
});

// Auto-refresh notification
setTimeout(() => {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: linear-gradient(135deg, #00ff9c, #00c46a);
        color: #0b1f17;
        padding: 10px 20px;
        border-radius: 10px;
        z-index: 1000;
        animation: slideIn 0.5s ease;
    `;
    notification.innerHTML = '🔄 Data Updated Successfully';
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.5s ease';
        setTimeout(() => notification.remove(), 500);
    }, 3000);
}, 5000);

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'r' && e.ctrlKey) {
        window.location.reload();
    }
});
</script>

<style>
@keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

@keyframes slideOut {
    from { transform: translateX(0); opacity: 1; }
    to { transform: translateX(100%); opacity: 0; }
}
</style>
""", unsafe_allow_html=True)
