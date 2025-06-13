import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 1. Generate simple synthetic time series (e.g., sine wave + noise)
def generate_sine_data(seq_len=1000):
    x = np.linspace(0, 50, seq_len)
    data = np.sin(x) + 0.1 * np.random.randn(seq_len)
    return data

series = generate_sine_data(2000)

# 2. Prepare sliding-window dataset
class SequenceDataset(Dataset):
    def __init__(self, data, input_len, target_len):
        self.data = data
        self.input_len = input_len
        self.target_len = target_len
        self.samples = []
        for i in range(len(data) - input_len - target_len):
            inp = data[i:i+input_len]
            tgt = data[i+input_len:i+input_len+target_len]
            self.samples.append((inp, tgt))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return torch.tensor(inp, dtype=torch.float32).unsqueeze(-1), \
               torch.tensor(tgt, dtype=torch.float32).unsqueeze(-1)

input_length = 20   # look back 20 timesteps
target_length = 1   # predict next timestep
dataset = SequenceDataset(series, input_length, target_length)

# Split into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# 3. Define a simple LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        # Take last timestep's output
        out = out[:, -1, :]
        return self.fc(out).unsqueeze(1)

model = SimpleLSTM(input_dim=1, hidden_dim=32, num_layers=1, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= train_size
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}")

# 5. Evaluate on test set and plot example
model.eval()
with torch.no_grad():
    xb, yb = next(iter(test_loader))
    preds = model(xb).squeeze().numpy()
    truths = yb.squeeze().numpy()

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# switch to eval mode
model.eval()

# gather all predictions & true values
preds, truths = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        out = model(xb)                    # (batch, 1, 1)
        preds.extend(out.squeeze().cpu().numpy())
        truths.extend(yb.squeeze().cpu().numpy())

preds = np.array(preds)
truths = np.array(truths)

# compute metrics
rmse = np.sqrt(mean_squared_error(truths, preds))
mae  = mean_absolute_error(truths, preds)
mard = np.mean(np.abs((preds - truths) / truths)) * 100  # in %

print(f"Test RMSE : {rmse:.2f} mg/dL")
print(f"Test MAE  : {mae:.2f} mg/dL")
print(f"Test MARD : {mard:.2f}%")


plt.plot(truths[:100], label='True')
plt.plot(preds[:100], label='Predicted')
plt.xlabel('Timestep')
plt.ylabel('Value')
plt.title('Simple LSTM: True vs Predicted on Test Sample')
plt.legend()
plt.show()