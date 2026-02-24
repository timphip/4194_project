"""
Day 9-10: 运行 MSE 和 LinEx 模型模拟
=======================================
Day 9:  MSE 模型 → 1000 天模拟 (baseline)
Day 10: LinEx 模型 → 1000 天模拟 (risk-aware)

预期: LinEx 模型因保守预测 → 更少故障, 可能更多维修
      但总成本更低 (避免了昂贵的故障惩罚)

运行方式: python day09_10_run_simulations.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from src.model import BiLSTM_RUL
from src.simulator import CostConfig, FleetSimulator
from src.analysis import plot_simulation_comparison

print("=" * 60)
print("Day 9-10: Fleet Simulation — MSE vs LinEx")
print("=" * 60)

# ---- Configuration ----
WINDOW_SIZE = 30
HIDDEN_SIZE = 64
FC_DIM = 128
DROPOUT = 0.3
N_DAYS = 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load RUL normalization factor
mse_data = np.load('checkpoints/mse_predictions.npz')
# Estimate rul_max from data
from src.data_generator import create_dataloaders
_, _, rul_max, _, _ = create_dataloaders(
    n_machines=100, window_size=WINDOW_SIZE,
    batch_size=64, train_ratio=0.8, seed=42
)

config = CostConfig(
    c_preventive=10.0,
    c_failure=100.0,
    c_waste=1.0,
    capacity=2,
    n_machines=20,
    safety_threshold=15,
)

# ---- Load models ----
print("\n[1] 加载模型...")

model_mse = BiLSTM_RUL(
    input_size=1, hidden_size=HIDDEN_SIZE,
    fc_dim=FC_DIM, dropout=DROPOUT
)
model_mse.load_state_dict(
    torch.load('checkpoints/model_mse.pt', weights_only=True,
               map_location=DEVICE)
)
model_mse = model_mse.to(DEVICE)
model_mse.eval()

model_linex = BiLSTM_RUL(
    input_size=1, hidden_size=HIDDEN_SIZE,
    fc_dim=FC_DIM, dropout=DROPOUT
)
model_linex.load_state_dict(
    torch.load('checkpoints/model_linex.pt', weights_only=True,
               map_location=DEVICE)
)
model_linex = model_linex.to(DEVICE)
model_linex.eval()

print("   模型加载成功 ✓")

# ============================================================
# Day 9: MSE Model Simulation
# ============================================================
print(f"\n{'=' * 40}")
print(f"Day 9: MSE 模型 — {N_DAYS} 天模拟")
print(f"{'=' * 40}")

sim_mse = FleetSimulator(
    config=config,
    model=model_mse,
    rul_max=rul_max,
    window_size=WINDOW_SIZE,
    device=DEVICE,
    seed=42
)

results_mse = sim_mse.run(n_days=N_DAYS, verbose=True)

print(f"\n   --- MSE 模型模拟结果 ---")
print(f"   总成本:     ${results_mse['total_cost']:.1f}")
print(f"   总故障:     {results_mse['total_failures']} 次")
print(f"   总维修:     {results_mse['total_maintenances']} 次")
print(f"   日均成本:   ${results_mse['avg_daily_cost']:.2f}")
print(f"   日均故障率: {results_mse['failure_rate']:.3f}")

# ============================================================
# Day 10: LinEx Model Simulation
# ============================================================
print(f"\n{'=' * 40}")
print(f"Day 10: LinEx 模型 — {N_DAYS} 天模拟")
print(f"{'=' * 40}")

sim_linex = FleetSimulator(
    config=config,
    model=model_linex,
    rul_max=rul_max,
    window_size=WINDOW_SIZE,
    device=DEVICE,
    seed=42  # Same seed for fair comparison
)

results_linex = sim_linex.run(n_days=N_DAYS, verbose=True)

print(f"\n   --- LinEx 模型模拟结果 ---")
print(f"   总成本:     ${results_linex['total_cost']:.1f}")
print(f"   总故障:     {results_linex['total_failures']} 次")
print(f"   总维修:     {results_linex['total_maintenances']} 次")
print(f"   日均成本:   ${results_linex['avg_daily_cost']:.2f}")
print(f"   日均故障率: {results_linex['failure_rate']:.3f}")

# ============================================================
# Comparison
# ============================================================
print(f"\n{'=' * 40}")
print(f"COMPARISON: MSE vs LinEx")
print(f"{'=' * 40}")

cost_diff = results_linex['total_cost'] - results_mse['total_cost']
fail_diff = results_linex['total_failures'] - results_mse['total_failures']
maint_diff = results_linex['total_maintenances'] - results_mse['total_maintenances']

print(f"   成本差异: {cost_diff:+.1f} ({'LinEx更便宜' if cost_diff < 0 else 'MSE更便宜'})")
print(f"   故障差异: {fail_diff:+d} ({'LinEx更少故障' if fail_diff < 0 else 'MSE更少故障'})")
print(f"   维修差异: {maint_diff:+d} ({'LinEx维修更多' if maint_diff > 0 else 'MSE维修更多'})")

if cost_diff < 0:
    print(f"\n   → LinEx 节省了 ${abs(cost_diff):.1f} "
          f"({abs(cost_diff)/results_mse['total_cost']*100:.1f}%)")
    print(f"   → 通过增加 {maint_diff} 次预防维修，减少 {abs(fail_diff)} 次故障")
    print(f"   → 每避免1次故障节省: ${abs(cost_diff)/max(abs(fail_diff),1):.1f}")

# ---- Plot comparison ----
print("\n[Plot] 生成对比图表...")
plot_simulation_comparison(results_mse, results_linex, save=True)

# ---- Save results ----
np.savez('checkpoints/simulation_results.npz',
         mse_cost=results_mse['total_cost'],
         mse_failures=results_mse['total_failures'],
         mse_maintenances=results_mse['total_maintenances'],
         mse_daily_costs=results_mse['daily_costs'],
         linex_cost=results_linex['total_cost'],
         linex_failures=results_linex['total_failures'],
         linex_maintenances=results_linex['total_maintenances'],
         linex_daily_costs=results_linex['daily_costs'])

print("\n✓ Day 9-10 完成!")
print("  图表已保存: figures/day09_10_simulation_comparison.png")
print("  结果已保存: checkpoints/simulation_results.npz")
