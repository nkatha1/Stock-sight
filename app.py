import streamlit as st
import torch
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

st.title("AAPL Stock Price Predictor")

SEQ_LENGTH = 30

# Load model
class StockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(SEQ_LENGTH, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = StockModel()
model.load_state_dict(torch.load("stock_model.pth"))
model.eval()

# Fetch data
data = yf.download("AAPL", period="1y")['Close'].values.reshape(-1,1)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
last_seq = torch.tensor(data_scaled[-SEQ_LENGTH:], dtype=torch.float32).view(1,-1)

# Predict
with torch.no_grad():
    pred_scaled = model(last_seq)
    pred_price = scaler.inverse_transform(pred_scaled.numpy())[0][0]

st.write(f"Predicted next day price for AAPL: ${pred_price:.2f}")