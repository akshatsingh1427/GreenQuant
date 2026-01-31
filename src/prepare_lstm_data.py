import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv("data/stock_data.csv")

# Clean
df = df.iloc[2:].reset_index(drop=True)
df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
for col in ["Close", "High", "Low", "Open", "Volume"]:
    df[col] = pd.to_numeric(df[col])

# Use Close price only (start simple)
close_prices = df["Close"].values.reshape(-1, 1)

# Scale
scaler = MinMaxScaler()
scaled = scaler.fit_transform(close_prices)

# Create sequences
X, y = [], []
window = 60

for i in range(window, len(scaled)):
    X.append(scaled[i-window:i])
    y.append(1 if scaled[i] > scaled[i-1] else 0)

X, y = np.array(X), np.array(y)

# Save for training
np.save("data/X_lstm.npy", X)
np.save("data/y_lstm.npy", y)

print("X shape:", X.shape)
print("y shape:", y.shape)
