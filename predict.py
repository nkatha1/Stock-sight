import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import sys

# Get CSV filename from command line
if len(sys.argv) < 2:
    print("Usage: python predict.py <stock_csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]

# Load data
data = pd.read_csv(csv_file)

if 'Close' not in data.columns:
    print("CSV must contain a 'Close' column")
    sys.exit(1)

close_prices = data['Close'].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(close_prices)

# Prepare last 30 days
seq_length = 30
last_seq = torch.tensor(data_scaled[-seq_length:], dtype=torch.float32).unsqueeze(0)

# Define Model
class StockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(seq_length, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = StockModel()
model.load_state_dict(torch.load("stock_model.pth"))
model.eval()

# Predict
with torch.no_grad():
    pred_scaled = model(last_seq)
    pred_price = scaler.inverse_transform(pred_scaled.numpy())[0, 0]

print(f"Predicted next day Close price: ${pred_price:.2f}")