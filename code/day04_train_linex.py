"""
Day 4: 训练 Risk-Aware (LinEx) 模型
======================================
验收标准：在同一张图上画出 MSE 和 LinEx 的预测线。
LinEx 的线应该明显跑在真实值的下方（低估寿命，保守）。

运行方式: python day04_train_linex.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data_generator import create_dataloaders
from src.train import create_and_train_model, get_predictions
from src.analysis import plot_prediction_comparison

print("=" * 60)
print("Day 4: Train Risk-Aware Model with LinEx Loss")
print("=" * 60)

# ---- Configuration (same as Day 3) ----
WINDOW_SIZE = 30
HIDDEN_SIZE = 64
FC_DIM = 128
DROPOUT = 0.3
N_EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 64
N_MACHINES = 100
SEED = 42
LINEX_A = 0.1  # 形状参数: 惩罚高估 RUL
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\n[Config] Device: {DEVICE}, LinEx a={LINEX_A}")

# ---- 1. Create dataloaders ----
print("\n[1] 准备数据...")
train_loader, test_loader, rul_max, test_health, test_rul = create_dataloaders(
    n_machines=N_MACHINES,
    window_size=WINDOW_SIZE,
    batch_size=BATCH_SIZE,
    train_ratio=0.8,
    seed=SEED
)

# ---- 2. Train LinEx model ----
print(f"\n[2] 训练 LinEx 模型 (a={LINEX_A})...")
model_linex, train_losses_linex, test_losses_linex = create_and_train_model(
    train_loader, test_loader, rul_max,
    loss_type='linex',
    linex_a=LINEX_A,
    hidden_size=HIDDEN_SIZE,
    fc_dim=FC_DIM,
    dropout=DROPOUT,
    n_epochs=N_EPOCHS,
    lr=LR,
    device=DEVICE,
    save_path='checkpoints/model_linex.pt',
    verbose=True
)

# ---- 3. Evaluate LinEx model ----
print("\n[3] 评估 LinEx 模型:")
preds_linex, trues = get_predictions(model_linex, test_loader, rul_max, DEVICE)

rmse_linex = np.sqrt(np.mean((preds_linex - trues) ** 2))
mae_linex = np.mean(np.abs(preds_linex - trues))
bias_linex = np.mean(preds_linex - trues)

print(f"   RMSE: {rmse_linex:.2f} cycles")
print(f"   MAE:  {mae_linex:.2f} cycles")
print(f"   Mean Bias: {bias_linex:.2f}")
print(f"   → LinEx 的 bias 应该为负数 (保守预测) ✓" if bias_linex < 0 else
      f"   → bias > 0, 可能需要增大 a 或更多 epochs")

# ---- 4. Load MSE results for comparison ----
print("\n[4] 对比 MSE vs LinEx:")
mse_data = np.load('checkpoints/mse_predictions.npz')
preds_mse = mse_data['preds']

bias_mse = np.mean(preds_mse - trues)
rmse_mse = np.sqrt(np.mean((preds_mse - trues) ** 2))

print(f"   {'Metric':<20} {'MSE':>10} {'LinEx':>10}")
print(f"   {'='*40}")
print(f"   {'RMSE':<20} {rmse_mse:>10.2f} {rmse_linex:>10.2f}")
print(f"   {'MAE':<20} {np.mean(np.abs(preds_mse - trues)):>10.2f} {mae_linex:>10.2f}")
print(f"   {'Mean Bias':<20} {bias_mse:>+10.2f} {bias_linex:>+10.2f}")

# 计算高估比例 (危险!)
overest_mse = np.mean(preds_mse > trues) * 100
overest_linex = np.mean(preds_linex > trues) * 100
print(f"   {'Overestimation %':<20} {overest_mse:>9.1f}% {overest_linex:>9.1f}%")
print()
print(f"   → LinEx 将高估率从 {overest_mse:.1f}% 降至 {overest_linex:.1f}%")
print(f"   → 这意味着更少的意外故障风险!")

# ---- 5. Plot comparison ----
print("\n[5] 生成对比图表...")
plot_prediction_comparison(preds_mse, preds_linex, trues, save=True)

# ---- 6. Save LinEx predictions ----
np.savez('checkpoints/linex_predictions.npz',
         preds=preds_linex, trues=trues)

print("\n✓ Day 4 完成!")
print("  LinEx 模型已保存: checkpoints/model_linex.pt")
print("  预测已保存: checkpoints/linex_predictions.npz")
print("  对比图已保存: figures/day04_prediction_comparison.png")
