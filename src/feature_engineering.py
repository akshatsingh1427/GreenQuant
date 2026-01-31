import pandas as pd
import numpy as np

# Load clean data
df = pd.read_csv("data/stock_data.csv")

# Remove garbage rows
df = df.iloc[2:].reset_index(drop=True)
df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

# Convert types
df["Date"] = pd.to_datetime(df["Date"])
for col in ["Close", "High", "Low", "Open", "Volume"]:
    df[col] = pd.to_numeric(df[col])

# =========================
# FEATURE ENGINEERING
# =========================

# Daily return
df["Return"] = df["Close"].pct_change()

# Moving averages
df["MA_5"] = df["Close"].rolling(window=5).mean()
df["MA_10"] = df["Close"].rolling(window=10).mean()

# Volatility
df["Volatility"] = df["Return"].rolling(window=5).std()

# =========================
# TARGET LABEL
# =========================

df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

# Drop NaN rows
df.dropna(inplace=True)

print(df.head())
print(df["Target"].value_counts())
