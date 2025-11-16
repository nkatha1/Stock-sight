import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import sys

# Get CSV filename from command line
if len(sys.argv) < 2:
    print("Usage: python train_model.py <stock_csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]

# Load data
data = pd.read_csv(csv_file)

# Keep only numeric columns
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
data = data[numeric_cols].dropna()

# Scale data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Prepare sequences
X = []
y = []
seq_length = 30

for i in range(seq_length, len(data_scaled)):
    X.append(data_scaled[i-seq_length:i, :])
    y.append(data_scaled[i, numeric_cols.index('Close')])  # Automatically find Close column

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

# Define Model
class StockModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = StockModel(seq_length * len(numeric_cols))
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs.squeeze(), y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

# Save model
torch.save(model.state_dict(), "stock_model.pth")
print("✔ Model trained and saved as stock_model.pth")