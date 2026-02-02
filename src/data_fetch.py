import yfinance as yf

ticker = "AAPL"

df = yf.download(
    ticker,
    start="2015-01-01",
    end="2024-01-01"
)

df.to_csv("data/stock_data.csv")
print("Data downloaded and saved successfully")

