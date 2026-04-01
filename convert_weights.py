import torch
import torch.nn as nn
import json
import pickle
import numpy as np
import os

# Definition of the model (just to load state dict easily, though we could just load the state_dict directly)
class CO2LSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, output_size=3, dropout=0.2):
        super(CO2LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

# Load state_dict
state_dict = torch.load('co2_lstm_1min.pth', map_location='cpu')

# In PyTorch, LSTM gates are IFGO (Input, Forget, Cell/Gate, Output).
# In TensorFlow, LSTM gates are IFCO (Input, Forget, Cell, Output). Their order matches!
# But PyTorch concatenates them as [W_i, W_f, W_c, W_o] along axis 0.
# TF concatenates them along axis 1 (for kernel: [input_dim, 4*units], recurrent_kernel: [units, 4*units], bias: [4*units]).
# Wait, PyTorch weight_ih_l0 is of shape [4*hidden_size, input_size].
# So transpose makes it [input_size, 4*hidden_size], which perfectly matches TF [input_dim, 4*units]!

out_weights = []

# Layer 1: LSTM l0
w_ih_l0 = state_dict['lstm.weight_ih_l0'].numpy()
w_hh_l0 = state_dict['lstm.weight_hh_l0'].numpy()
b_ih_l0 = state_dict['lstm.bias_ih_l0'].numpy()
b_hh_l0 = state_dict['lstm.bias_hh_l0'].numpy()

out_weights.append(w_ih_l0.T.tolist())               # kernel: [7, 512]
out_weights.append(w_hh_l0.T.tolist())               # recurrent_kernel: [128, 512]
out_weights.append((b_ih_l0 + b_hh_l0).tolist())     # bias: [512]

# Layer 2: LSTM l1
w_ih_l1 = state_dict['lstm.weight_ih_l1'].numpy()
w_hh_l1 = state_dict['lstm.weight_hh_l1'].numpy()
b_ih_l1 = state_dict['lstm.bias_ih_l1'].numpy()
b_hh_l1 = state_dict['lstm.bias_hh_l1'].numpy()

out_weights.append(w_ih_l1.T.tolist())               # kernel: [128, 512]
out_weights.append(w_hh_l1.T.tolist())               # recurrent_kernel: [128, 512]
out_weights.append((b_ih_l1 + b_hh_l1).tolist())     # bias: [512]

# FC 1
w_fc0 = state_dict['fc.0.weight'].numpy()
b_fc0 = state_dict['fc.0.bias'].numpy()

out_weights.append(w_fc0.T.tolist())                 # kernel: [128, 32]
out_weights.append(b_fc0.tolist())                   # bias: [32]

# FC 2 (index 3 because index 1 is ReLU, index 2 is Dropout)
w_fc3 = state_dict['fc.3.weight'].numpy()
b_fc3 = state_dict['fc.3.bias'].numpy()

out_weights.append(w_fc3.T.tolist())                 # kernel: [32, 3]
out_weights.append(b_fc3.tolist())                   # bias: [3]

# Write to JSON
with open('model/base_weights.json', 'w') as f:
    json.dump(out_weights, f)

print(f"Saved {len(out_weights)} tensors to model/base_weights.json")
print("Shapes printed to verify:")
for i, w in enumerate(out_weights):
    print(f"Tensor {i}: {np.array(w).shape}")

# Load scalers
with open('feature_scaler_1min.pkl', 'rb') as f:
    feature_scaler = pickle.load(f)

with open('target_scaler_1min.pkl', 'rb') as f:
    target_scaler = pickle.load(f)

print("FEATURE_MIN:", feature_scaler.data_min_.tolist())
print("FEATURE_MAX:", feature_scaler.data_max_.tolist())
print("TARGET_MIN:", target_scaler.data_min_[0])
print("TARGET_MAX:", target_scaler.data_max_[0])
