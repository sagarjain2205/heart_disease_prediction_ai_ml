import pandas as pd
import numpy as np
import joblib, os
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. Load Data ---
df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Scaling (Essential for PyTorch) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# --- 3. XGBoost Implementation ---
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
xgb_model.fit(X_train_scaled, y_train)
xgb_model.save_model("models/xgboost_model.json")

# --- 4. PyTorch Neural Network ---
class HeartNet(nn.Module):
    def __init__(self, input_dim):
        super(HeartNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# Training Loop
model = HeartNet(X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

X_tensor = torch.FloatTensor(X_train_scaled)
y_tensor = torch.FloatTensor(y_train.values).view(-1, 1)

for epoch in range(100):
    optimizer.zero_grad()
    loss = criterion(model(X_tensor), y_tensor)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "models/pytorch_model.pth")
print("✅ Advanced Models (XGBoost & PyTorch) saved in /models")