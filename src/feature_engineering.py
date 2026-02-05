import pandas as pd
import numpy as np


df = pd.read_csv("data/stock_data.csv")

df = df.iloc[2:].reset_index(drop=True)
df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]


df["Date"] = pd.to_datetime(df["Date"])
for col in ["Close", "High", "Low", "Open", "Volume"]:
    df[col] = pd.to_numeric(df[col])




df["Return"] = df["Close"].pct_change()


df["MA_5"] = df["Close"].rolling(window=5).mean()
df["MA_10"] = df["Close"].rolling(window=10).mean()

df["Volatility"] = df["Return"].rolling(window=5).std()



df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)


df.dropna(inplace=True)

print(df.head())
print(df["Target"].value_counts())

