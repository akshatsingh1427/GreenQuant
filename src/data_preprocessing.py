import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/stock_data.csv")

# 🔹 Remove first two garbage rows
df = df.iloc[2:].reset_index(drop=True)

# 🔹 Rename columns properly
df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

# 🔹 Convert data types
df["Date"] = pd.to_datetime(df["Date"])
df["Close"] = pd.to_numeric(df["Close"])
df["High"] = pd.to_numeric(df["High"])
df["Low"] = pd.to_numeric(df["Low"])
df["Open"] = pd.to_numeric(df["Open"])
df["Volume"] = pd.to_numeric(df["Volume"])

print(df.head())
print(df.dtypes)

# 🔹 Plot closing price
plt.figure(figsize=(10,5))
plt.plot(df["Date"], df["Close"])
plt.title("AAPL Stock Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()
