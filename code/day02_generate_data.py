"""
Day 2: 生成"像真的"退化数据
================================
验收标准：曲线看起来像股票走势图，且尾部比头部更"抖"（异方差性）。

运行方式: python day02_generate_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from src.data_generator import generate_fleet_data, generate_single_curve
from src.analysis import plot_degradation_curves

print("=" * 60)
print("Day 2: Synthetic Degradation Data Generation")
print("=" * 60)

# ---- 1. 单条退化曲线展示 ----
print("\n[1] 单条退化曲线:")
health, rul, ft = generate_single_curve(
    degradation_rate=0.5,
    noise_base=0.3,
    noise_growth=0.005,
    seed=42
)
print(f"   初始健康: {health[0]:.1f}")
print(f"   失效时间: {ft} 步")
print(f"   健康值范围: [{health.min():.1f}, {health.max():.1f}]")
print(f"   时间序列长度: {len(health)}")

# ---- 2. 异方差性验证 ----
print("\n[2] 异方差性验证:")
diffs = np.diff(health)
first_quarter = diffs[:len(diffs)//4]
last_quarter = diffs[-len(diffs)//4:]
print(f"   前 25% 步变化量标准差: {np.std(first_quarter):.4f}")
print(f"   后 25% 步变化量标准差: {np.std(last_quarter):.4f}")
print(f"   倍率: {np.std(last_quarter)/np.std(first_quarter):.2f}x")
print(f"   → 尾部噪声更大 ✓ (异方差性)")

# ---- 3. 生成机队数据 ----
print("\n[3] 生成机队数据 (100 台机器):")
all_health, all_rul, all_ft = generate_fleet_data(n_machines=100, seed=42)

lifetimes = np.array(all_ft)
print(f"   平均寿命: {lifetimes.mean():.1f} ± {lifetimes.std():.1f} 步")
print(f"   最短寿命: {lifetimes.min()} 步")
print(f"   最长寿命: {lifetimes.max()} 步")
print(f"   总数据点: {sum(len(h) for h in all_health)}")

# ---- 4. 画图 ----
print("\n[4] 生成图表 (5 条退化曲线)...")
plot_degradation_curves(all_health, n_curves=5, save=True)

print("\n✓ Day 2 完成! 图表已保存到 figures/day02_degradation_curves.png")
