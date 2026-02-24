"""
Day 3: 搭建 Baseline (MSE) 模型
=================================
验收标准：在测试集画图，预测线应该穿过真实数据点的中间。

运行方式: python day03_train_mse.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data_generator import create_dataloaders
from src.model import BiLSTM_RUL
from src.train import create_and_train_model, get_predictions

print("=" * 60)
print("Day 3: Train Baseline Bi-LSTM with MSE Loss")
print("=" * 60)

# ---- Configuration ----
WINDOW_SIZE = 30
HIDDEN_SIZE = 64
FC_DIM = 128
DROPOUT = 0.3
N_EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 64
N_MACHINES = 100
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\n[Config] Device: {DEVICE}, Epochs: {N_EPOCHS}, Window: {WINDOW_SIZE}")

# ---- 1. Create dataloaders ----
print("\n[1] 准备数据...")
train_loader, test_loader, rul_max, test_health, test_rul = create_dataloaders(
    n_machines=N_MACHINES,
    window_size=WINDOW_SIZE,
    batch_size=BATCH_SIZE,
    train_ratio=0.8,
    seed=SEED
)
print(f"   训练样本: {len(train_loader.dataset)}")
print(f"   测试样本: {len(test_loader.dataset)}")
print(f"   RUL 最大值 (归一化因子): {rul_max:.1f}")

# ---- 2. Train MSE model ----
print("\n[2] 训练 MSE Baseline 模型...")
os.makedirs('checkpoints', exist_ok=True)

model_mse, train_losses, test_losses = create_and_train_model(
    train_loader, test_loader, rul_max,
    loss_type='mse',
    hidden_size=HIDDEN_SIZE,
    fc_dim=FC_DIM,
    dropout=DROPOUT,
    n_epochs=N_EPOCHS,
    lr=LR,
    device=DEVICE,
    save_path='checkpoints/model_mse.pt',
    verbose=True
)

# ---- 3. Evaluate ----
print("\n[3] 评估结果:")
preds_mse, trues = get_predictions(model_mse, test_loader, rul_max, DEVICE)

rmse = np.sqrt(np.mean((preds_mse - trues) ** 2))
mae = np.mean(np.abs(preds_mse - trues))
bias = np.mean(preds_mse - trues)

print(f"   RMSE: {rmse:.2f} cycles")
print(f"   MAE:  {mae:.2f} cycles")
print(f"   Mean Bias: {bias:.2f} (正=高估, 负=低估)")
print(f"   → MSE 模型的 bias ≈ 0，因为 MSE 是对称的 (风险中性)")

# ---- 4. Plot results ----
print("\n[4] 生成图表...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Training curve
ax = axes[0]
ax.plot(train_losses, label='Train Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Training Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# Test RMSE curve
ax = axes[1]
ax.plot(test_losses, label='Test RMSE', color='orange')
ax.set_xlabel('Epoch')
ax.set_ylabel('RMSE (cycles)')
ax.set_title('Test RMSE vs Epoch')
ax.legend()
ax.grid(True, alpha=0.3)

# Prediction scatter
ax = axes[2]
n_show = min(500, len(trues))
idx = np.random.choice(len(trues), n_show, replace=False)
sort_idx = np.argsort(trues[idx])

ax.plot(trues[idx][sort_idx], trues[idx][sort_idx], 'k--', linewidth=2, label='Perfect')
ax.scatter(trues[idx][sort_idx], preds_mse[idx][sort_idx], 
           alpha=0.3, s=10, c='blue', label='MSE Prediction')
ax.set_xlabel('True RUL')
ax.set_ylabel('Predicted RUL')
ax.set_title(f'MSE Baseline: RMSE={rmse:.1f}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/day03_mse_baseline.png', dpi=150)
plt.show()

# ---- 5. Save predictions for later comparison ----
np.savez('checkpoints/mse_predictions.npz', 
         preds=preds_mse, trues=trues)

print("\n✓ Day 3 完成!")
print("  模型已保存: checkpoints/model_mse.pt")
print("  预测已保存: checkpoints/mse_predictions.npz")
print("  图表已保存: figures/day03_mse_baseline.png")
