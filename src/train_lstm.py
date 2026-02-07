import numpy as np
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os

TICKER = "AAPL"
START = "2015-01-01"
END = "2024-01-01"
WINDOW = 60
EPOCHS = 10
BATCH_SIZE = 32

os.makedirs("models", exist_ok=True)

df = yf.download(TICKER, start=START, end=END)
close = df["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(close)

X, y = [], []
for i in range(WINDOW, len(scaled)):
    X.append(scaled[i-WINDOW:i])
    y.append(1 if scaled[i] > scaled[i-1] else 0)

X = np.array(X)
y = np.array(y)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WINDOW, 1)),
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X, y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

model.save_weights("models/lstm_weights.weights.h5")

print("✅ Weights saved correctly")

