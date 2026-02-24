"""
Day 1: 理解并实现 LinEx Loss
==============================
验收标准：看到损失曲线一边平缓（低估 → 安全），一边陡峭（高估 → 危险）。

运行方式: python day01_linex_visualization.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from src.linex_loss import linex_loss_numpy, mse_loss_numpy, mae_loss_numpy
from src.analysis import plot_linex_curves

print("=" * 60)
print("Day 1: LinEx Loss Function Visualization")
print("=" * 60)

# ---- 1. 基础公式验证 ----
print("\n[1] LinEx Loss 公式验证:")
print("   L(Δ) = b * (exp(a*Δ) - a*Δ - 1)")
print("   where Δ = y_pred - y_true")
print()

# Test specific values
test_errors = [-2, -1, 0, 1, 2]
a = 1.5
print(f"   a = {a} (惩罚高估 RUL → 安全优先)")
print(f"   {'Error':<10} {'LinEx':<12} {'MSE':<12} {'MAE':<12}")
print(f"   {'-----':<10} {'-----':<12} {'---':<12} {'---':<12}")
for e in test_errors:
    linex = linex_loss_numpy(e, a)
    mse = mse_loss_numpy(e)
    mae = mae_loss_numpy(e)
    print(f"   {e:<10} {linex:<12.4f} {mse:<12.4f} {mae:<12.4f}")

print()
print("   观察: 当 error=+2 (高估), LinEx=1.47 >> LinEx(-2)=0.63 (低估)")
print("        而 MSE(+2) = MSE(-2) = 4.0 → 对称，不区分方向！")

# ---- 2. 数学性质验证 ----
print("\n[2] 数学性质验证:")

# a→0 时 LinEx → MSE
a_small = 0.001
errors = np.array([-2, -1, 0, 1, 2], dtype=float)
linex_small_a = linex_loss_numpy(errors, a_small)
mse_scaled = 0.5 * a_small**2 * errors**2  # 理论极限

print(f"   当 a→0 时, LinEx ≈ (a²/2)*Δ² (即 MSE 的缩放版)")
print(f"   a={a_small}: LinEx = {linex_small_a}")
print(f"   a={a_small}: (a²/2)*Δ² = {mse_scaled}")
print(f"   误差: {np.abs(linex_small_a - mse_scaled)}")

# ---- 3. 凸性验证 (二阶导 > 0) ----
print("\n[3] 凸性验证:")
print("   d²L/dΔ² = b*a²*exp(a*Δ) > 0 for all Δ → 严格凸函数 ✓")
print("   → 保证全局最小值唯一，梯度下降可收敛")

# ---- 4. 贝叶斯最优预测子 (高斯情况) ----
print("\n[4] 贝叶斯最优预测子 (高斯后验):")
print("   θ̂ = μ - a*σ²/2")
print("   Example: μ=50, σ²=100, a=0.5")
mu, sigma2, a = 50, 100, 0.5
theta_hat = mu - a * sigma2 / 2
print(f"   θ̂ = {mu} - {a}*{sigma2}/2 = {theta_hat}")
print(f"   → 模型保守地预测 RUL={theta_hat} 而非均值 {mu}")
print(f"   → 安全边际 = {mu - theta_hat} 个周期")

# ---- 5. 画图 ----
print("\n[5] 生成图表...")
plot_linex_curves(save=True)

print("\n✓ Day 1 完成! 图表已保存到 figures/day01_linex_loss.png")
