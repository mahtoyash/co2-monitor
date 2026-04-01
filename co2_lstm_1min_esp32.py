#!/usr/bin/env python
# coding: utf-8

# # CO2 LSTM Model — 1-Min Interval (ESP32 Ready)
# ### 3 Predictions: 10 min, 30 min, 60 min ahead
# **Cold start: 24 min** | ESP32 @ 1-min → direct prediction

# ## Cell 1 — Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')
if torch.cuda.is_available():
    print(f'GPU    : {torch.cuda.get_device_name(0)}')
print('Libraries loaded!')


# ## Cell 2 — Load Dataset

# In[ ]:


# =============================================
# Apna file path yahan daalo
# =============================================
FILE_PATH = 'dataset123.xlsx'
# =============================================

df = pd.read_excel(FILE_PATH)

df = df.rename(columns={
    'wendu(℃)'  : 'Temperature',
    'shidu(RH)' : 'Humidity',
    'sum'       : 'Occupancy',
    'CO2(ppm)'  : 'CO2'
})

df = df[['Temperature', 'Humidity', 'Occupancy', 'CO2']].copy()

print(f'Shape   : {df.shape}')
print(f'Columns : {df.columns.tolist()}')
print(f'Missing :\n{df.isnull().sum()}')
print(f'\nCO2 range: {df["CO2"].min()} – {df["CO2"].max()} ppm')
df.head()


# ## Cell 3 — 10-min → 1-min Interpolation
# > Dataset 10-min interval ka hai. ESP32 1-min pe data bhejega — isliye linear interpolation se 1-min data banate hain.
# > **6,336 rows → 63,351 rows**

# In[ ]:


# Timestamps assign karo (paper: 2022-10-17 00:00, 10-min interval)
timestamps_10min = pd.date_range(start='2022-10-17 00:00:00', periods=len(df), freq='10min')
df.index = timestamps_10min

# 1-min pe resample + linear interpolate
df_1min = df.resample('1min').interpolate(method='linear')

# Occupancy round karo — person count hai, float nahi chahiye
df_1min['Occupancy'] = df_1min['Occupancy'].round().astype(int)
df_1min['Occupancy'] = df_1min['Occupancy'].clip(lower=0)

print(f'Original  (10-min) : {len(df):,} rows')
print(f'Resampled  (1-min) : {len(df_1min):,} rows')
print(f'CO2 range          : {df_1min["CO2"].min():.0f} – {df_1min["CO2"].max():.0f} ppm')
print(f'Occupancy range    : {df_1min["Occupancy"].min()} – {df_1min["Occupancy"].max()}')


# ## Cell 4 — Time Features Generate Karo

# In[ ]:


df_1min = df_1min.reset_index()
df_1min = df_1min.rename(columns={'index': 'date'})

df_1min['hour']        = df_1min['date'].dt.hour
df_1min['day_of_week'] = df_1min['date'].dt.dayofweek
df_1min['minute']      = df_1min['date'].dt.minute

df_1min = df_1min.drop(columns=['date'])

print(f'Start : 2022-10-17 00:00')
print(f'End   : {pd.date_range(start="2022-10-17", periods=len(df_1min), freq="1min")[-1]}')
print(f'Columns: {df_1min.columns.tolist()}')
df_1min.head(12)


# ## Cell 5 — Data Check

# In[ ]:


print('CO2 stats:')
print(df_1min['CO2'].describe())

fig, axes = plt.subplots(2, 2, figsize=(16, 6))

axes[0,0].plot(df_1min['CO2'].values[:500], color='steelblue', linewidth=0.8)
axes[0,0].set_title('CO2 — first 500 readings (1-min)')
axes[0,0].set_ylabel('CO2 (ppm)')

axes[0,1].plot(df_1min['Temperature'].values[:500], color='tomato', linewidth=0.8)
axes[0,1].set_title('Temperature')
axes[0,1].set_ylabel('°C')

axes[1,0].plot(df_1min['Humidity'].values[:500], color='seagreen', linewidth=0.8)
axes[1,0].set_title('Humidity')
axes[1,0].set_ylabel('RH%')

axes[1,1].plot(df_1min['Occupancy'].values[:500], color='darkorange', linewidth=0.8)
axes[1,1].set_title('Occupancy')
axes[1,1].set_ylabel('People count')

plt.tight_layout()
plt.show()


# ## Cell 6 — Normalize

# In[ ]:


FEATURE_COLS = ['Temperature', 'Humidity', 'Occupancy', 'CO2', 'hour', 'day_of_week', 'minute']
TARGET_COL   = 'CO2'

feature_scaler = MinMaxScaler()
target_scaler  = MinMaxScaler()

df_scaled = df_1min.copy()
df_scaled[FEATURE_COLS] = feature_scaler.fit_transform(df_1min[FEATURE_COLS])
target_scaler.fit(df_1min[[TARGET_COL]])

with open('feature_scaler_1min.pkl', 'wb') as f:
    pickle.dump(feature_scaler, f)
with open('target_scaler_1min.pkl', 'wb') as f:
    pickle.dump(target_scaler, f)

print('Normalization done!')
print('\nScaled range check (should be [0, 1]):')
print(df_scaled[FEATURE_COLS].describe().loc[['min', 'max']])


# ## Cell 7 — Sequences Banao
# > **Sequence = 24 steps = 24 min lookback** | Horizons: 10, 30, 60 steps = 10, 30, 60 min
# > **ESP32 cold start: sirf 24 min!** ✅

# In[ ]:


SEQUENCE_LEN = 24    # 24 min lookback @ 1-min interval
PREDICT_10   = 10    # 10 min ahead
PREDICT_30   = 30    # 30 min ahead
PREDICT_60   = 60    # 60 min ahead

data_array = df_scaled[FEATURE_COLS].values
co2_array  = df_scaled[TARGET_COL].values

X, y = [], []

for i in range(SEQUENCE_LEN, len(data_array) - PREDICT_60):
    X.append(data_array[i - SEQUENCE_LEN : i])
    y.append([
        co2_array[i + PREDICT_10],
        co2_array[i + PREDICT_30],
        co2_array[i + PREDICT_60]
    ])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print(f'X shape : {X.shape}  → [samples, {SEQUENCE_LEN} steps, {len(FEATURE_COLS)} features]')
print(f'y shape : {y.shape}  → [samples, 3 horizons]')
print(f'\nESP32 cold start : {SEQUENCE_LEN} min = {SEQUENCE_LEN} readings ✅')


# ## Cell 8 — Train/Test Split

# In[ ]:


split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f'Train samples : {len(X_train):,}')
print(f'Test samples  : {len(X_test):,}')
print(f'Split ratio   : 80/20')


# ## Cell 9 — PyTorch Dataset + DataLoader

# In[ ]:


class CO2Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

BATCH_SIZE = 64

train_dataset = CO2Dataset(X_train, y_train)
test_dataset  = CO2Dataset(X_test,  y_test)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f'Train batches : {len(train_loader)}')
print(f'Test batches  : {len(test_loader)}')


# ## Cell 10 — LSTM Model
# > **Architecture:** 2 LSTM layers | 128 hidden units | Dropout 0.2 | FC head (128→32→3)

# In[ ]:


class CO2LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=3, dropout=0.2):
        super(CO2LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out    = lstm_out[:, -1, :]
        return self.fc(last_out)

INPUT_SIZE = len(FEATURE_COLS)   # 7
model      = CO2LSTMModel(input_size=INPUT_SIZE, hidden_size=128, num_layers=2).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f'Input size : {INPUT_SIZE} features')
print(f'Parameters : {total_params:,}')
print(f'Model size : ~{total_params * 4 / 1024:.1f} KB')
print()
print(model)


# ## Cell 11 — Training

# In[ ]:


EPOCHS   = 200
LR       = 0.001
patience = 20

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

train_losses = []
test_losses  = []
best_loss    = float('inf')
no_improve   = 0

print(f'Training started | Device: {device}')
print('-' * 55)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss   = criterion(output, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output     = model(X_batch)
            test_loss += criterion(output, y_batch).item()

    avg_train = train_loss / len(train_loader)
    avg_test  = test_loss  / len(test_loader)

    train_losses.append(avg_train)
    test_losses.append(avg_test)
    scheduler.step(avg_test)

    if avg_test < best_loss:
        best_loss  = avg_test
        no_improve = 0
        torch.save(model.state_dict(), 'co2_lstm_1min.pth')
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1:3d}/{EPOCHS}] | Train: {avg_train:.6f} | Test: {avg_test:.6f}')

print(f'\nBest Test Loss : {best_loss:.6f} (MSE)')
print('Model saved → co2_lstm_1min.pth')


# ## Cell 12 — Loss Plot

# In[ ]:


plt.figure(figsize=(12, 4))
plt.plot(train_losses, label='Train Loss', color='steelblue', linewidth=1.5)
plt.plot(test_losses,  label='Test Loss',  color='tomato',    linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training vs Test Loss — CO2 LSTM (1-min)')
plt.legend()
plt.tight_layout()
plt.show()


# ## Cell 13 — Accuracy Metrics
# > MAE, RMSE, MAPE, R², Accuracy (±10%)

# In[ ]:


model.load_state_dict(torch.load('co2_lstm_1min.pth'))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch.to(device)).cpu().numpy()
        all_preds.append(preds)
        all_true.append(y_batch.numpy())

all_preds = np.concatenate(all_preds)
all_true  = np.concatenate(all_true)

def inv_co2(arr):
    return target_scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

labels = ['10 min (10 steps)', '30 min (30 steps)', '60 min (60 steps)']

print('=' * 58)
print(f'  {"Horizon":<22} {"MAE":>6} {"RMSE":>7} {"MAPE":>7} {"R²":>8} {"Acc±10%":>9}')
print('=' * 58)

for i, label in enumerate(labels):
    pred_ppm = inv_co2(all_preds[:, i])
    true_ppm = inv_co2(all_true[:,  i])

    mae    = np.mean(np.abs(pred_ppm - true_ppm))
    rmse   = np.sqrt(np.mean((pred_ppm - true_ppm) ** 2))
    mape   = np.mean(np.abs((true_ppm - pred_ppm) / (true_ppm + 1e-8))) * 100
    ss_res = np.sum((true_ppm - pred_ppm) ** 2)
    ss_tot = np.sum((true_ppm - np.mean(true_ppm)) ** 2)
    r2     = 1 - (ss_res / (ss_tot + 1e-8))
    acc    = np.mean(np.abs((true_ppm - pred_ppm) / (true_ppm + 1e-8)) <= 0.10) * 100

    print(f'  {label:<22} {mae:>6.2f} {rmse:>7.2f} {mape:>6.2f}% {r2:>8.4f} {acc:>8.1f}%')

print('=' * 58)


# ## Cell 14 — Prediction vs Actual Plot

# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(18, 4))
colors  = ['steelblue', 'seagreen', 'darkorange']
samples = 200

for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
    pred_ppm = inv_co2(all_preds[:samples, i])
    true_ppm = inv_co2(all_true[:samples,  i])

    ax.plot(true_ppm, label='Actual',    color='black', linewidth=1.2)
    ax.plot(pred_ppm, label='Predicted', color=color,   linewidth=1.2, linestyle='--')
    ax.set_title(f'CO2 — {label}')
    ax.set_xlabel('Sample')
    ax.set_ylabel('CO2 (ppm)')
    ax.legend()

plt.suptitle('Actual vs Predicted CO2 — 1-min Model (ESP32 Ready)', fontsize=13, y=1.02)
plt.tight_layout()
plt.show()


# ## Cell 15 — Spike Detection

# In[ ]:


SPIKE_THRESHOLD = 1000   # ppm

model.eval()
with torch.no_grad():
    sample_input = torch.tensor(X_test[:5], dtype=torch.float32).to(device)
    raw_preds    = model(sample_input).cpu().numpy()

print(f'Spike threshold : {SPIKE_THRESHOLD} ppm')
print(f'\n{"Sample":<8} {"10 min":>10} {"30 min":>10} {"60 min":>10} {"SPIKE ALERT":>14}')
print('-' * 55)

for s in range(5):
    p10 = inv_co2(raw_preds[s:s+1, 0])[0]
    p30 = inv_co2(raw_preds[s:s+1, 1])[0]
    p60 = inv_co2(raw_preds[s:s+1, 2])[0]
    spike = '⚠️  YES' if any(p > SPIKE_THRESHOLD for p in [p10, p30, p60]) else '✅  NO'
    print(f'{s+1:<8} {p10:>10.1f} {p30:>10.1f} {p60:>10.1f} {spike:>14}')


# ## Cell 16 — Config Save

# In[ ]:


config = {
    'model_name'         : 'CO2LSTMModel_1min_ESP32',
    'dataset'            : 'dataset123.xlsx interpolated to 1-min',
    'sampling_interval'  : '1 min (interpolated from 10-min)',
    'esp32_interval'     : '1 min',
    'cold_start_minutes' : SEQUENCE_LEN,
    'sequence_len'       : SEQUENCE_LEN,
    'feature_cols'       : FEATURE_COLS,
    'target_col'         : TARGET_COL,
    'predict_steps'      : [PREDICT_10, PREDICT_30, PREDICT_60],
    'predict_labels'     : ['10 min', '30 min', '60 min'],
    'spike_threshold'    : SPIKE_THRESHOLD,
    'input_size'         : INPUT_SIZE,
    'hidden_size'        : 128,
    'num_layers'         : 2,
    'output_size'        : 3,
    'dropout'            : 0.2,
    'timestamp_start'    : '2022-10-17 00:00:00',
    'feature_scaler_min' : feature_scaler.data_min_.tolist(),
    'feature_scaler_max' : feature_scaler.data_max_.tolist(),
    'target_scaler_min'  : target_scaler.data_min_.tolist(),
    'target_scaler_max'  : target_scaler.data_max_.tolist(),
}

with open('model_config_1min.json', 'w') as f:
    json.dump(config, f, indent=2)

print('Saved artifacts:')
print('  co2_lstm_1min.pth        → model weights')
print('  model_config_1min.json   → architecture + config')
print('  feature_scaler_1min.pkl  → input scaler')
print('  target_scaler_1min.pkl   → output scaler')
print(f'\nESP32 cold start : {SEQUENCE_LEN} min ✅')
print('Done ✅')

