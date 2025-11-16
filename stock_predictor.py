import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
TICKER = "AAPL"
SEQ_LENGTH = 30
EPOCHS = 50
LR = 0.001

# ----------------------------
# Download stock data
# ----------------------------
print("✔ Downloading AAPL data...")
data = yf.download(TICKER, start="2015-01-01", end="2025-01-01")['Close'].values
data = data.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Prepare sequences
X, y = [], []
for i in range(SEQ_LENGTH, len(data_scaled)):
    X.append(data_scaled[i-SEQ_LENGTH:i, 0])
    y.append(data_scaled[i, 0])

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

# ----------------------------
# Define Model
# ----------------------------
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

# ----------------------------
# Train Model
# ----------------------------
print("Training model...")
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs.squeeze(), y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.6f}")

# Save model
torch.save(model.state_dict(), "stock_model.pth")
print("✔ Model trained and saved as stock_model.pth")

# ----------------------------
# Predict Next Day
# ----------------------------
last_seq = torch.tensor(data_scaled[-SEQ_LENGTH:], dtype=torch.float32)
last_seq = last_seq.view(1, -1)  # reshape to [1, 30]

model.eval()
with torch.no_grad():
    pred_scaled = model(last_seq)
    pred_price = scaler.inverse_transform(pred_scaled.detach().numpy())[0][0]
    print(f"Predicted next day price for {TICKER}: ${pred_price:.2f}")

# ----------------------------
# Optional: Plot last 100 days + prediction
# ----------------------------
plt.plot(data[-100:], label="Actual Price")
plt.plot(range(len(data)-1, len(data)), [pred_price], 'ro', label="Next Day Prediction")
plt.title(f"{TICKER} Stock Price")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()