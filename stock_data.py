import yfinance as yf
import pandas as pd

# Download Apple stock data
data = yf.download("AAPL", start="2015-01-01", end="2025-01-01")

data.to_csv("AAPL.csv")
print("✔ Data downloaded and saved as AAPL.csv")