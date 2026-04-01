#!/usr/bin/env python
# coding: utf-8

# # CO2 LSTM Model — Classroom Dataset
# 
# **Dataset:** University Classroom, China | Oct–Nov 2022 | 10-min interval

# ## Cell 1 — Libraries

# In[3]:


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
print('Libraries loaded')


# ## Cell 2 — Load Dataset

# In[4]:


FILE_PATH = 'dataset123.xlsx'

df = pd.read_excel(FILE_PATH)

# Rename Chinese columns to English
df = df.rename(columns={
    'wendu(℃)'    : 'Temperature',
    'shidu(RH)'    : 'Humidity',
    'sum'          : 'Occupancy',
    'CO2(ppm)'     : 'CO2'
})

# Keep only required columns
df = df[['Temperature', 'Humidity', 'Occupancy', 'CO2']].copy()

print(f'Shape   : {df.shape}')
print(f'Columns : {df.columns.tolist()}')
print(f'Missing :\n{df.isnull().sum()}')
print(f'\nCO2 range: {df["CO2"].min()} – {df["CO2"].max()} ppm')
df.head()


# ## Cell 3 — Timestamp Generation
# > Dataset collected: **17 Oct 2022 00:00 → 30 Nov 2022**, every **10 min**. No timestamp column in Excel — generating programmatically from paper metadata.

# In[5]:


# Generate timestamp from paper: start = 2022-10-17 00:00, freq = 10 min
START_TIME = '2022-10-17 00:00:00'
timestamps = pd.date_range(start=START_TIME, periods=len(df), freq='10min')

df['date'] = timestamps
df = df.sort_values('date').reset_index(drop=True)

# Extract time features
df['hour']        = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek   # 0=Monday, 6=Sunday
df['minute']      = df['date'].dt.minute

# Verify date range
print(f'Start : {df["date"].min()}')
print(f'End   : {df["date"].max()}')
print(f'Total : {len(df)} rows ({len(df)/(144):.1f} days)')
print(f'\nColumns after timestamp engineering:')
print(df.drop(columns=["date"]).columns.tolist())
df.drop(columns=['date']).head(10)


# ## Cell 4 — Data Check

# In[6]:


print('CO2 stats:')
print(df['CO2'].describe())
print(f'\nCO2 change per reading:')
print(df['CO2'].diff().abs().describe())

fig, axes = plt.subplots(2, 2, figsize=(16, 6))

axes[0,0].plot(df['CO2'].values[:500], color='steelblue', linewidth=0.8)
axes[0,0].set_title('CO2 — first 500 readings')
axes[0,0].set_ylabel('CO2 (ppm)')
axes[0,0].set_xlabel('Row')

axes[0,1].plot(df['Temperature'].values[:500], color='tomato', linewidth=0.8)
axes[0,1].set_title('Temperature — first 500 readings')
axes[0,1].set_ylabel('°C')

axes[1,0].plot(df['Humidity'].values[:500], color='seagreen', linewidth=0.8)
axes[1,0].set_title('Humidity — first 500 readings')
axes[1,0].set_ylabel('RH%')

axes[1,1].plot(df['Occupancy'].values[:500], color='darkorange', linewidth=0.8)
axes[1,1].set_title('Occupancy — first 500 readings')
axes[1,1].set_ylabel('People count')

plt.tight_layout()
plt.show()


# ## Cell 5 — Normalize

# In[7]:


FEATURE_COLS = ['Temperature', 'Humidity', 'Occupancy', 'CO2', 'hour', 'day_of_week', 'minute']
TARGET_COL   = 'CO2'

feature_scaler = MinMaxScaler()
target_scaler  = MinMaxScaler()

df_scaled = df.copy()
df_scaled[FEATURE_COLS] = feature_scaler.fit_transform(df[FEATURE_COLS])
target_scaler.fit(df[[TARGET_COL]])

# Save scalers
with open('feature_scaler_classroom.pkl', 'wb') as f:
    pickle.dump(feature_scaler, f)
with open('target_scaler_classroom.pkl', 'wb') as f:
    pickle.dump(target_scaler, f)

print('Normalization done!')
print('\nScaled range check (should be [0, 1]):')
print(df_scaled[FEATURE_COLS].describe().loc[['min', 'max']])


# ## Cell 6 — Sequencce
# > **Sequence length = 18 steps = 3 hours** | Horizons: 1 step (10 min), 3 steps (30 min), 6 steps (60 min)

# In[8]:


SEQUENCE_LEN = 24    # 3 hours lookback @ 10-min intervals
PREDICT_10   = 1     # 10 min ahead  (1 step)
PREDICT_30   = 3     # 30 min ahead  (3 steps)
PREDICT_60   = 6     # 60 min ahead  (6 steps)

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

print(f'X shape : {X.shape}  → [samples, seq_len, features]')
print(f'y shape : {y.shape}  → [samples, 3 horizons]')


# ## Cell 7 — Train/Test Split

# In[9]:


split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f'Train samples : {len(X_train)}')
print(f'Test samples  : {len(X_test)}')
print(f'Split ratio   : 80/20')


# ## Cell 8 — PyTorch Dataset + DataLoader

# In[10]:


class CO2Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

BATCH_SIZE = 32   # Small dataset -> smaller batch

train_dataset = CO2Dataset(X_train, y_train)
test_dataset  = CO2Dataset(X_test,  y_test)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f'Train batches : {len(train_loader)}')
print(f'Test batches  : {len(test_loader)}')


# ## Cell 9 — LSTM Model
# > **Architecture:** 2 LSTM layers | 128 hidden units | Dropout 0.2 | FC head (128→32→3)

# In[11]:


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


# ## Cell 10 — Training

# In[23]:


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
    # --- Train ---
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

    # --- Evaluate ---
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

    # --- Early stopping + save best ---
    if avg_test < best_loss:
        best_loss  = avg_test
        no_improve = 0
        torch.save(model.state_dict(), 'co2_lstm_classroom.pth')
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1:3d}/{EPOCHS}] | Train: {avg_train:.6f} | Test: {avg_test:.6f}')

print(f'\nBest Test Loss : {best_loss:.6f} (MSE)')
print('Model saved → co2_lstm_classroom.pth')


# ## Cell 11 — Loss Plot

# In[24]:


plt.figure(figsize=(12, 4))
plt.plot(train_losses, label='Train Loss', color='steelblue', linewidth=1.5)
plt.plot(test_losses,  label='Test Loss',  color='tomato',    linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training vs Test Loss — Classroom CO2 LSTM')
plt.legend()
plt.tight_layout()
plt.show()


# ## Cell 12 — Accuracy Metrics
# > MAE, RMSE, MAPE, R², Accuracy (±10%)

# In[12]:


model.load_state_dict(torch.load('co2_lstm_classroom.pth'))
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

labels   = ['10 min (1 step)', '30 min (3 steps)', '60 min (6 steps)']
horizons = [PREDICT_10, PREDICT_30, PREDICT_60]

print('=' * 55)
print(f'  {"Horizon":<20} {"MAE":>7} {"RMSE":>8} {"MAPE":>8} {"R²":>8} {"Acc±10%":>9}')
print('=' * 55)

for i, label in enumerate(labels):
    pred_ppm = inv_co2(all_preds[:, i])
    true_ppm = inv_co2(all_true[:,  i])

    mae  = np.mean(np.abs(pred_ppm - true_ppm))
    rmse = np.sqrt(np.mean((pred_ppm - true_ppm) ** 2))
    mape = np.mean(np.abs((true_ppm - pred_ppm) / (true_ppm + 1e-8))) * 100
    ss_res = np.sum((true_ppm - pred_ppm) ** 2)
    ss_tot = np.sum((true_ppm - np.mean(true_ppm)) ** 2)
    r2   = 1 - (ss_res / (ss_tot + 1e-8))
    acc  = np.mean(np.abs((true_ppm - pred_ppm) / (true_ppm + 1e-8)) <= 0.10) * 100

    print(f'  {label:<20} {mae:>7.2f} {rmse:>8.2f} {mape:>7.2f}% {r2:>8.4f} {acc:>8.1f}%')

print('=' * 55)


# In[ ]:





# In[13]:


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Helper ──────────────────────────────────────────────────
def inv_co2(arr):
    return target_scaler.inverse_transform(arr.reshape(-1,1)).flatten()

labels      = ['10 min (10 steps)', '30 min (30 steps)', '60 min (60 steps)']
colors_pred = ['steelblue', 'seagreen', 'darkorange']
horizons    = [0, 1, 2]

# Pre-compute all ppm values
preds_ppm = [inv_co2(all_preds[:, i]) for i in horizons]
trues_ppm = [inv_co2(all_true[:,  i]) for i in horizons]
errors    = [p - t for p, t in zip(preds_ppm, trues_ppm)]

# ── Metrics ─────────────────────────────────────────────────
mae_list, rmse_list, r2_list, acc_list, mape_list = [], [], [], [], []
for i in horizons:
    p, t  = preds_ppm[i], trues_ppm[i]
    mae   = np.mean(np.abs(p - t))
    rmse  = np.sqrt(np.mean((p - t)**2))
    mape  = np.mean(np.abs((t - p) / (t + 1e-8))) * 100
    ss_r  = np.sum((t - p)**2)
    ss_t  = np.sum((t - np.mean(t))**2)
    r2    = 1 - ss_r / (ss_t + 1e-8)
    acc   = np.mean(np.abs((t - p) / (t + 1e-8)) <= 0.10) * 100
    mae_list.append(mae); rmse_list.append(rmse)
    r2_list.append(r2);   acc_list.append(acc); mape_list.append(mape)


# GRAPH 1 — Actual vs Predicted (2000 samples, spike region)

start, n = 0, len(preds_ppm[0])
fig, axes = plt.subplots(3, 1, figsize=(18, 10))
for i, (ax, label, color) in enumerate(zip(axes, labels, colors_pred)):
    ax.plot(trues_ppm[i][start:start+n], label='Actual',    color='black',  linewidth=1.0)
    ax.plot(preds_ppm[i][start:start+n], label='Predicted', color=color,    linewidth=1.0, linestyle='--')
    ax.set_title(f'Actual vs Predicted CO2 — {label}', fontsize=11)
    ax.set_ylabel('CO2 (ppm)'); ax.set_xlabel('Sample')
    ax.legend(); ax.grid(alpha=0.3)
plt.suptitle('Actual vs Predicted CO2 — All Horizons', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('graph1_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show(); print("Graph 1 saved")

# GRAPH 2 — Scatter Plot: Actual vs Predicted
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, (ax, label, color) in enumerate(zip(axes, labels, colors_pred)):
    p, t = preds_ppm[i], trues_ppm[i]
    ax.scatter(t, p, alpha=0.2, s=3, color=color)
    mn, mx = min(t.min(), p.min()), max(t.max(), p.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect fit')
    ax.set_title(f'Scatter — {label}', fontsize=10)
    ax.set_xlabel('Actual CO2 (ppm)'); ax.set_ylabel('Predicted CO2 (ppm)')
    ax.text(0.05, 0.92, f'R² = {r2_list[i]:.4f}', transform=ax.transAxes,
            fontsize=10, color='black', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.legend(); ax.grid(alpha=0.3)
plt.suptitle('Scatter Plot: Actual vs Predicted CO2', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('graph2_scatter.png', dpi=150, bbox_inches='tight')
plt.show(); print("Graph 2 saved")

# GRAPH 3 — Error Distribution (Histogram)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for i, (ax, label, color) in enumerate(zip(axes, labels, colors_pred)):
    ax.hist(errors[i], bins=80, color=color, edgecolor='white', alpha=0.85)
    ax.axvline(0,  color='black', linewidth=1.5, linestyle='--', label='Zero error')
    ax.axvline( mae_list[i], color='red',  linewidth=1.2, linestyle=':', label=f'+MAE={mae_list[i]:.1f}')
    ax.axvline(-mae_list[i], color='red',  linewidth=1.2, linestyle=':')
    ax.set_title(f'Error Distribution — {label}', fontsize=10)
    ax.set_xlabel('Prediction Error (ppm)'); ax.set_ylabel('Frequency')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle('Prediction Error Distribution — All Horizons', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('graph3_error_distribution.png', dpi=150, bbox_inches='tight')
plt.show(); print("Graph 3 saved")

# GRAPH 4 — Metrics Bar Chart (MAE, RMSE, MAPE, R², Acc)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
x = np.arange(3); short = ['10 min', '30 min', '60 min']

# MAE & RMSE
bars1 = axes[0].bar(x - 0.2, mae_list,  0.4, label='MAE',  color='steelblue',  alpha=0.85)
bars2 = axes[0].bar(x + 0.2, rmse_list, 0.4, label='RMSE', color='lightsteelblue', alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(short)
axes[0].set_ylabel('ppm'); axes[0].set_title('MAE & RMSE (ppm)')
axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)
for bar in bars1: axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
for bar in bars2: axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

# R²
bars3 = axes[1].bar(x, r2_list, 0.5, color=['#2ecc71','#f39c12','#e74c3c'], alpha=0.85)
axes[1].set_xticks(x); axes[1].set_xticklabels(short)
axes[1].set_ylim(0, 1.1); axes[1].axhline(0.9, color='green', linestyle='--', linewidth=1, label='0.9 target')
axes[1].set_ylabel('R²'); axes[1].set_title('R² Score per Horizon')
axes[1].legend(); axes[1].grid(axis='y', alpha=0.3)
for bar in bars3: axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)

# Accuracy ±10%
bars4 = axes[2].bar(x, acc_list, 0.5, color=['#2ecc71','#f39c12','#e74c3c'], alpha=0.85)
axes[2].set_xticks(x); axes[2].set_xticklabels(short)
axes[2].set_ylim(80, 102); axes[2].axhline(90, color='red', linestyle='--', linewidth=1, label='90% target')
axes[2].set_ylabel('%'); axes[2].set_title('Accuracy (±10%) per Horizon')
axes[2].legend(); axes[2].grid(axis='y', alpha=0.3)
for bar in bars4: axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)

plt.suptitle('Model Performance Metrics — All Horizons', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('graph4_metrics_bar.png', dpi=150, bbox_inches='tight')
plt.show(); print("Graph 4 saved")

# ════════════════════════════════════════════════════════════
# GRAPH 5 — Residual Plot (Error over time)
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(18, 8))
for i, (ax, label, color) in enumerate(zip(axes, labels, colors_pred)):
    ax.plot(errors[i][start:start+n], color=color, linewidth=0.7, alpha=0.85)
    ax.axhline(0,             color='black', linewidth=1.2, linestyle='--')
    ax.axhline( mae_list[i],  color='red',   linewidth=1.0, linestyle=':', label=f'+MAE={mae_list[i]:.1f} ppm')
    ax.axhline(-mae_list[i],  color='red',   linewidth=1.0, linestyle=':')
    ax.set_title(f'Residual (Predicted − Actual) — {label}', fontsize=10)
    ax.set_ylabel('Error (ppm)'); ax.set_xlabel('Sample')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle('Residual Plot — All Horizons', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('graph5_residual.png', dpi=150, bbox_inches='tight')
plt.show(); print("Graph 5 saved")

# ════════════════════════════════════════════════════════════
# GRAPH 6 — Spike Detection Visualization
# ════════════════════════════════════════════════════════════
SPIKE_THRESHOLD = 1000
spike_start, spike_n = 0, len(preds_ppm[0])
spike_mask = np.any(
    np.column_stack([preds_ppm[i] > SPIKE_THRESHOLD for i in horizons]), axis=1
)

fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(trues_ppm[0], color='black', linewidth=0.8, label='Actual CO2 (10 min true)', alpha=0.9)
ax.plot(preds_ppm[0], color='steelblue', linewidth=0.8, linestyle='--', label='Predicted (10 min)', alpha=0.8)
ax.axhline(SPIKE_THRESHOLD, color='red', linewidth=1.5, linestyle='--', label=f'Spike threshold ({SPIKE_THRESHOLD} ppm)')

# Shade spike zones
in_spike = False
for j in range(len(spike_mask)):
    if spike_mask[j] and not in_spike:
        spike_start_j = j; in_spike = True
    elif not spike_mask[j] and in_spike:
        ax.axvspan(spike_start_j, j, alpha=0.25, color='red')
        in_spike = False
if in_spike:
    ax.axvspan(spike_start_j, len(spike_mask), alpha=0.25, color='red', label='Spike Alert Zone')

ax.set_title('CO2 Spike Detection — Full Test Set', fontsize=12, fontweight='bold')
ax.set_xlabel('Sample'); ax.set_ylabel('CO2 (ppm)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('graph6_spike_detection.png', dpi=150, bbox_inches='tight')
plt.show(); print("Graph 6 save")

# ════════════════════════════════════════════════════════════
# GRAPH 7 — Cumulative Error (CDF)
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
for i, (label, color) in enumerate(zip(labels, colors_pred)):
    abs_err = np.abs(errors[i])
    sorted_err = np.sort(abs_err)
    cdf = np.arange(1, len(sorted_err)+1) / len(sorted_err)
    ax.plot(sorted_err, cdf * 100, color=color, linewidth=2, label=label)
ax.axvline(20, color='gray', linestyle='--', linewidth=1, label='20 ppm mark')
ax.set_xlabel('Absolute Error (ppm)'); ax.set_ylabel('Cumulative % of Samples')
ax.set_title('Cumulative Error Distribution (CDF) — All Horizons', fontsize=12, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0, 150)
plt.tight_layout(); plt.savefig('graph7_cdf.png', dpi=150, bbox_inches='tight')
plt.show(); print("Graph 7 saved")

print("\n" + "="*58)
print(f"  {'Horizon':<22} {'MAE':>6} {'RMSE':>7} {'MAPE':>7} {'R²':>8} {'Acc±10%':>9}")
print("="*58)
for i, label in enumerate(labels):
    print(f"  {label:<22} {mae_list[i]:>6.2f} {rmse_list[i]:>7.2f} {mape_list[i]:>6.2f}% {r2_list[i]:>8.4f} {acc_list[i]:>8.1f}%")
print("="*58)
print("\nSaved: graph1 to graph7")


# In[ ]:





# ## Cell 13 — Prediction vs Actual Plot

# In[26]:


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

plt.suptitle('Actual vs Predicted CO2 — Classroom Dataset', fontsize=13, y=1.02)
plt.tight_layout()
plt.show()


# ## Cell 14 — Spike Detection
# > Alert triggered when predicted CO₂ exceeds threshold (default: 1000 ppm)

# In[13]:


SPIKE_THRESHOLD = 1000   # ppm — standard indoor CO2 alert level (paper reference)

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
    spike = 'YES' if any(p > SPIKE_THRESHOLD for p in [p10, p30, p60]) else 'NO'
    print(f'{s+1:<8} {p10:>10.1f} {p30:>10.1f} {p60:>10.1f} {spike:>14}')


# ## Cell 15 — Config Save

# In[14]:


config = {
    'model_name'         : 'CO2LSTMModel_Classroom',
    'dataset'            : 'dataset123.xlsx (University Classroom, China, Oct-Nov 2022)',
    'sequence_len'       : SEQUENCE_LEN,
    'feature_cols'       : FEATURE_COLS,
    'target_col'         : TARGET_COL,
    'predict_steps'      : [PREDICT_10, PREDICT_30, PREDICT_60],
    'predict_labels'     : ['10 min', '30 min', '60 min'],
    'sampling_interval'  : '10 min',
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

with open('model_config_classroom.json', 'w') as f:
    json.dump(config, f, indent=2)

print('Saved artifacts:')
print('  co2_lstm_classroom.pth        → model weights')
print('  model_config_classroom.json   → architecture + config')
print('  feature_scaler_classroom.pkl  → input scaler')
print('  target_scaler_classroom.pkl   → output scaler')
print()


# In[ ]:




