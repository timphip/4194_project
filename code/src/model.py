"""
Day 3: Bi-LSTM Model Architecture
===================================
Minimalist Bi-LSTM + 3 FC layers for RUL prediction.

Architecture:
    Input  → [batch, window_size, 1]   (health index readings)
    BiLSTM → [batch, window_size, 2*hidden_size]
    Take last step → [batch, 2*hidden_size]
    FC1 + ReLU → [batch, fc_dim]
    FC2 + ReLU → [batch, fc_dim]
    FC3        → [batch, 1]  (predicted RUL, normalized)

Reference: Noussis et al. (2024) minimalist Bi-LSTM approach
"""

import torch
import torch.nn as nn


class BiLSTM_RUL(nn.Module):
    """
    Bidirectional LSTM followed by 3 fully-connected layers.
    
    Args:
        input_size:  number of features per time step (1 = health index)
        hidden_size: LSTM hidden state dimension (each direction)
        fc_dim:      dimension of intermediate FC layers
        dropout:     dropout rate (used in FC layers and optionally in LSTM)
        num_layers:  number of LSTM layers (stacked)
    """
    
    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        fc_dim=128,
        dropout=0.3,
        num_layers=1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 3 Fully-Connected layers
        lstm_output_dim = hidden_size * 2  # bidirectional → 2x hidden
        
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(fc_dim, fc_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(fc_dim // 2, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_size]
        
        Returns:
            out: [batch_size, 1] predicted RUL (normalized)
        """
        # LSTM forward pass
        # lstm_out: [batch, seq_len, 2*hidden_size]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the output at the last time step
        # lstm_out[:, -1, :] → [batch, 2*hidden_size]
        last_output = lstm_out[:, -1, :]
        
        # Pass through FC layers
        out = self.fc_layers(last_output)
        
        return out
    
    def predict_with_dropout(self, x, n_samples=50):
        """
        Monte Carlo Dropout inference for uncertainty quantification.
        Keep dropout active during inference and run multiple forward passes.
        
        Args:
            x: [batch_size, seq_len, input_size]
            n_samples: number of MC forward passes
        
        Returns:
            mean_pred: [batch_size, 1] mean prediction
            std_pred:  [batch_size, 1] prediction std (epistemic uncertainty)
            all_preds: [n_samples, batch_size, 1] all predictions
        """
        self.train()  # Keep dropout active
        
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(x)
                preds.append(out)
        
        all_preds = torch.stack(preds, dim=0)  # [n_samples, batch, 1]
        mean_pred = all_preds.mean(dim=0)       # [batch, 1]
        std_pred = all_preds.std(dim=0)          # [batch, 1]
        
        return mean_pred, std_pred, all_preds
