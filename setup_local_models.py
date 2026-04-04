import os
import torch
import torch.nn as nn
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

os.makedirs("models", exist_ok=True)

# 1. Sklearn Mock Pipeline (StandardScaler + Logistic Regression)
X = np.array([[100.0, 1.0, 1], [200.0, 2.0, 2], [5000.0, 23.0, 1]])
y = np.array([0, 0, 1])

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
pipeline.fit(X, y)
joblib.dump(pipeline, "models/fraud.pkl")
print("Saved models/fraud.pkl")

# 2. PyTorch Mock Model (CNN)
class DummyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 224 * 224, 2)
    def forward(self, x):
        return self.fc(self.flatten(x))

model = DummyCNN()
model.eval()
dummy_input = torch.randn(1, 3, 224, 224) 
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("models/cnn.pt")
print("Saved models/cnn.pt")
