import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---- Parameters ----
TICKER = "AAPL"
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
SEQ_LENGTH = 30
EPOCHS = 50
LR = 0.001

# ---- Download Stock Data ----
print("✔ Downloading stock data...")
data = yf.download(TICKER, start=START_DATE, end=END_DATE)["Close"]
data = data.values.reshape(-1, 1)

# ---- Scale Data ----
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# ---- Prepare Sequences ----
X, y = [], []
for i in range(SEQ_LENGTH, len(data_scaled)):
    X.append(data_scaled[i-SEQ_LENGTH:i, 0])
    y.append(data_scaled[i, 0])

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

# ---- Define Model ----
class StockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(SEQ_LENGTH, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = StockModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---- Train Model ----
print("Training model...")
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs.squeeze(), y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.6f}")

# Save model
torch.save(model.state_dict(), "stock_model.pth")
print("✔ Model trained and saved as stock_model.pth")

# ---- Predict Next Day ----
last_30_days = torch.tensor(data_scaled[-SEQ_LENGTH:], dtype=torch.float32).unsqueeze(0)
pred_scaled = model(last_30_days)
pred_price = scaler.inverse_transform(pred_scaled.detach().numpy())[0][0]
print(f"Predicted next day price for {TICKER}: ${pred_price:.2f}")

# ---- Plot Actual vs Predicted ----
predicted_all = model(X).detach().numpy()
predicted_all = scaler.inverse_transform(predicted_all)
plt.figure(figsize=(12,6))
plt.plot(data, label="Actual Price")
plt.plot(range(SEQ_LENGTH, len(data)), predicted_all, label="Predicted Price")
plt.title(f"{TICKER} Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()