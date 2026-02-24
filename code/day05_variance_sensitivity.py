"""
Day 5: 验证"方差敏感性"
==========================
验收标准：看到一个趋势 —— X 越大（越不确定），Y 越负（越保守）。

理论依据：θ̂_Bayes = μ - a*σ²/2
    → 不确定性(σ²)越大，LinEx的保守偏移越大

运行方式: python day05_variance_sensitivity.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from src.data_generator import create_dataloaders
from src.model import BiLSTM_RUL
from src.train import get_predictions, get_predictions_with_uncertainty
from src.analysis import plot_variance_sensitivity

print("=" * 60)
print("Day 5: Variance Sensitivity Analysis")
print("=" * 60)

# ---- Configuration ----
WINDOW_SIZE = 30
HIDDEN_SIZE = 64
FC_DIM = 128
DROPOUT = 0.3
N_MACHINES = 100
BATCH_SIZE = 64
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- 1. Load data and models ----
print("\n[1] 加载数据和模型...")
train_loader, test_loader, rul_max, test_health, test_rul = create_dataloaders(
    n_machines=N_MACHINES,
    window_size=WINDOW_SIZE,
    batch_size=BATCH_SIZE,
    train_ratio=0.8,
    seed=SEED
)

# Load LinEx model
model_linex = BiLSTM_RUL(
    input_size=1, hidden_size=HIDDEN_SIZE,
    fc_dim=FC_DIM, dropout=DROPOUT
)
model_linex.load_state_dict(
    torch.load('checkpoints/model_linex.pt', weights_only=True,
               map_location=DEVICE)
)
model_linex = model_linex.to(DEVICE)

# Load MSE model for comparison
model_mse = BiLSTM_RUL(
    input_size=1, hidden_size=HIDDEN_SIZE,
    fc_dim=FC_DIM, dropout=DROPOUT
)
model_mse.load_state_dict(
    torch.load('checkpoints/model_mse.pt', weights_only=True,
               map_location=DEVICE)
)
model_mse = model_mse.to(DEVICE)

print("   模型加载成功 ✓")

# ---- 2. Get predictions ----
print("\n[2] 生成预测值...")
preds_linex, trues = get_predictions(model_linex, test_loader, rul_max, DEVICE)
preds_mse, _ = get_predictions(model_mse, test_loader, rul_max, DEVICE)

# ---- 3. Monte Carlo Dropout uncertainty ----
print("\n[3] Monte Carlo Dropout 不确定性估计 (50 次前向传播)...")
mean_preds_mc, std_preds_mc, trues_mc = get_predictions_with_uncertainty(
    model_linex, test_loader, rul_max, n_mc=50, device=DEVICE
)

print(f"   预测均值范围: [{mean_preds_mc.min():.1f}, {mean_preds_mc.max():.1f}]")
print(f"   预测标准差范围: [{std_preds_mc.min():.2f}, {std_preds_mc.max():.2f}]")

# ---- 4. Variance-Bias scatter for LinEx model ----
print("\n[4] 生成 LinEx 方差-偏差散点图...")
plot_variance_sensitivity(preds_linex, trues, test_health, WINDOW_SIZE, save=True)

# ---- 5. MC Dropout Variance vs Bias ----
print("\n[5] MC Dropout 方差 vs 偏差分析:")

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LinEx: MC uncertainty vs bias
ax = axes[0]
mc_bias = mean_preds_mc - trues_mc
mc_var = std_preds_mc ** 2

ax.scatter(mc_var, mc_bias, alpha=0.2, s=5, c='indianred')
if len(mc_var) > 10:
    z = np.polyfit(mc_var, mc_bias, 1)
    p = np.poly1d(z)
    var_range = np.linspace(mc_var.min(), mc_var.max(), 100)
    ax.plot(var_range, p(var_range), 'k-', linewidth=2,
            label=f'Trend: slope={z[0]:.2f}')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('MC Dropout Variance (σ²)')
ax.set_ylabel('Prediction Bias (Pred - True)')
ax.set_title('LinEx: MC Uncertainty vs Bias\n(Theory: bias = -a*σ²/2)')
ax.legend()
ax.grid(True, alpha=0.3)

# MSE: for comparison  
ax = axes[1]
mean_preds_mc_mse, std_preds_mc_mse, trues_mc_mse = get_predictions_with_uncertainty(
    model_mse, test_loader, rul_max, n_mc=50, device=DEVICE
)
mc_bias_mse = mean_preds_mc_mse - trues_mc_mse
mc_var_mse = std_preds_mc_mse ** 2

ax.scatter(mc_var_mse, mc_bias_mse, alpha=0.2, s=5, c='steelblue')
if len(mc_var_mse) > 10:
    z = np.polyfit(mc_var_mse, mc_bias_mse, 1)
    p = np.poly1d(z)
    var_range = np.linspace(mc_var_mse.min(), mc_var_mse.max(), 100)
    ax.plot(var_range, p(var_range), 'k-', linewidth=2,
            label=f'Trend: slope={z[0]:.2f}')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('MC Dropout Variance (σ²)')
ax.set_ylabel('Prediction Bias (Pred - True)')
ax.set_title('MSE: MC Uncertainty vs Bias\n(No systematic shift expected)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/day05_mc_dropout_analysis.png', dpi=150)
plt.show()

# ---- 6. 理论验证: bias ≈ -a*σ²/2 ----
print("\n[6] 理论验证 bias ≈ -a*σ²/2:")
a = 0.1  # LinEx parameter used in training
theoretical_bias = -a * np.mean(mc_var) / 2
actual_bias = np.mean(mc_bias)

print(f"   LinEx a = {a}")
print(f"   Mean MC Variance: {np.mean(mc_var):.4f}")
print(f"   Theoretical bias = -a*mean(σ²)/2 = {theoretical_bias:.4f}")
print(f"   Actual mean bias = {actual_bias:.4f}")
print(f"   (Note: DNN bias is not purely Bayesian; this is an approximation)")

print("\n✓ Day 5 完成!")
print("  图表已保存: figures/day05_variance_sensitivity.png")
print("  图表已保存: figures/day05_mc_dropout_analysis.png")
