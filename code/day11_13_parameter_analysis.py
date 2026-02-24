"""
Day 11-13: 参数扫描、成本微笑曲线与瓶颈分析
===============================================
Day 11: 参数扫描 — a 从 0.0 到 2.0
Day 12: 绘制"成本微笑曲线" (U-shape)
Day 13: 瓶颈分析 — 改变 Capacity K

运行方式: python day11_13_parameter_analysis.py

注意: 此脚本需要较长时间 (为每个 a 值重新训练模型并运行模拟)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from tqdm import tqdm

from src.data_generator import create_dataloaders
from src.model import BiLSTM_RUL
from src.train import create_and_train_model
from src.simulator import CostConfig, FleetSimulator
from src.analysis import plot_cost_smile_curve, plot_capacity_analysis

print("=" * 60)
print("Day 11-13: Parameter Sweep & Bottleneck Analysis")
print("=" * 60)

# ---- Configuration ----
WINDOW_SIZE = 30
HIDDEN_SIZE = 64
FC_DIM = 128
DROPOUT = 0.3
N_EPOCHS_SWEEP = 60       # Fewer epochs for sweep efficiency
N_DAYS_SIM = 500           # Fewer days for sweep efficiency
BATCH_SIZE = 64
N_MACHINES = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- Prepare Data ----
print("\n[0] 准备数据...")
train_loader, test_loader, rul_max, _, _ = create_dataloaders(
    n_machines=N_MACHINES,
    window_size=WINDOW_SIZE,
    batch_size=BATCH_SIZE,
    train_ratio=0.8,
    seed=42
)

# ============================================================
# Day 11: Parameter Sweep
# ============================================================
print(f"\n{'=' * 40}")
print("Day 11: 参数扫描 — Finding the Golden a")
print(f"{'=' * 40}")

# a values to sweep
a_values = np.array([0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])

print(f"\n   Sweeping a ∈ {a_values}")
print(f"   Each: train {N_EPOCHS_SWEEP} epochs + simulate {N_DAYS_SIM} days")
print(f"   Total configurations: {len(a_values)}")

config_base = CostConfig(
    c_preventive=10.0,
    c_failure=100.0,
    c_waste=1.0,
    capacity=2,
    n_machines=20,
    safety_threshold=15,
)

sweep_costs = []
sweep_failures = []
sweep_maintenances = []

for i, a_val in enumerate(a_values):
    print(f"\n   --- [{i+1}/{len(a_values)}] a = {a_val:.2f} ---")
    
    # For a=0 (or very small), use MSE
    if a_val < 0.001:
        loss_type = 'mse'
        linex_a = 0.0
    else:
        loss_type = 'linex'
        linex_a = a_val
    
    # Train model
    model, _, _ = create_and_train_model(
        train_loader, test_loader, rul_max,
        loss_type=loss_type,
        linex_a=linex_a,
        hidden_size=HIDDEN_SIZE,
        fc_dim=FC_DIM,
        dropout=DROPOUT,
        n_epochs=N_EPOCHS_SWEEP,
        lr=1e-3,
        device=DEVICE,
        save_path=None,
        verbose=False
    )
    
    # Run simulation
    sim = FleetSimulator(
        config=config_base,
        model=model,
        rul_max=rul_max,
        window_size=WINDOW_SIZE,
        device=DEVICE,
        seed=42
    )
    
    results = sim.run(n_days=N_DAYS_SIM, verbose=False)
    sweep_costs.append(results['total_cost'])
    sweep_failures.append(results['total_failures'])
    sweep_maintenances.append(results['total_maintenances'])
    
    print(f"      Cost: ${results['total_cost']:.1f}, "
          f"Failures: {results['total_failures']}, "
          f"Maint: {results['total_maintenances']}")

sweep_costs = np.array(sweep_costs)
sweep_failures = np.array(sweep_failures)
sweep_maintenances = np.array(sweep_maintenances)

# ============================================================
# Day 12: Cost Smile Curve
# ============================================================
print(f"\n{'=' * 40}")
print("Day 12: 绘制成本微笑曲线")
print(f"{'=' * 40}")

plot_cost_smile_curve(a_values, sweep_costs, save=True)

# ---- Additional analysis plot ----
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Cost vs a
ax = axes[0]
ax.plot(a_values, sweep_costs, 'o-', color='steelblue', linewidth=2)
min_idx = np.argmin(sweep_costs)
ax.plot(a_values[min_idx], sweep_costs[min_idx], 'r*', markersize=15)
ax.set_xlabel('LinEx a')
ax.set_ylabel('Total Cost')
ax.set_title('Cost Smile Curve')
ax.grid(True, alpha=0.3)

# Failures vs a
ax = axes[1]
ax.plot(a_values, sweep_failures, 'o-', color='indianred', linewidth=2)
ax.set_xlabel('LinEx a')
ax.set_ylabel('Total Failures')
ax.set_title('Failures vs Risk Aversion')
ax.grid(True, alpha=0.3)

# Maintenances vs a
ax = axes[2]
ax.plot(a_values, sweep_maintenances, 'o-', color='forestgreen', linewidth=2)
ax.set_xlabel('LinEx a')
ax.set_ylabel('Total Maintenances')
ax.set_title('Preventive Maintenances vs Risk Aversion')
ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/day11_sweep_details.png', dpi=150)
plt.show()

# ============================================================
# Day 13: Bottleneck Analysis (Varying Capacity K)
# ============================================================
print(f"\n{'=' * 40}")
print("Day 13: 瓶颈分析 — 改变 Capacity K")
print(f"{'=' * 40}")

capacity_values = [1, 2, 3, 5]
a_values_cap = np.array([0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])

sweep_results = {}

for cap_val in capacity_values:
    print(f"\n   --- Capacity K = {cap_val} ---")
    
    config_cap = CostConfig(
        c_preventive=10.0,
        c_failure=100.0,
        c_waste=1.0,
        capacity=cap_val,
        n_machines=20,
        safety_threshold=15,
    )
    
    costs_for_cap = []
    
    for a_val in a_values_cap:
        # Train model
        if a_val < 0.001:
            loss_type = 'mse'
            linex_a = 0.0
        else:
            loss_type = 'linex'
            linex_a = a_val
        
        model, _, _ = create_and_train_model(
            train_loader, test_loader, rul_max,
            loss_type=loss_type,
            linex_a=linex_a,
            hidden_size=HIDDEN_SIZE,
            fc_dim=FC_DIM,
            dropout=DROPOUT,
            n_epochs=N_EPOCHS_SWEEP,
            lr=1e-3,
            device=DEVICE,
            save_path=None,
            verbose=False
        )
        
        # Simulate
        sim = FleetSimulator(
            config=config_cap,
            model=model,
            rul_max=rul_max,
            window_size=WINDOW_SIZE,
            device=DEVICE,
            seed=42
        )
        
        results = sim.run(n_days=N_DAYS_SIM, verbose=False)
        costs_for_cap.append(results['total_cost'])
    
    sweep_results[cap_val] = {
        'a_values': a_values_cap,
        'costs': np.array(costs_for_cap)
    }
    
    min_idx = np.argmin(costs_for_cap)
    print(f"      Optimal a = {a_values_cap[min_idx]:.2f}, "
          f"Min Cost = ${costs_for_cap[min_idx]:.1f}")

plot_capacity_analysis(capacity_values, sweep_results, save=True)

# ---- Save all results ----
np.savez('checkpoints/sweep_results.npz',
         a_values=a_values,
         costs=sweep_costs,
         failures=sweep_failures,
         maintenances=sweep_maintenances,
         capacity_values=np.array(capacity_values))

print(f"\n{'=' * 40}")
print("分析总结")
print(f"{'=' * 40}")
print(f"\n   最优 a (K=2):  {a_values[np.argmin(sweep_costs)]:.2f}")
print(f"   最低成本 (K=2): ${sweep_costs.min():.1f}")
print(f"   MSE 成本 (a≈0): ${sweep_costs[0]:.1f}")
if sweep_costs[0] > 0:
    print(f"   成本节省:        {(1 - sweep_costs.min()/sweep_costs[0])*100:.1f}%")

for cap_val in capacity_values:
    costs = sweep_results[cap_val]['costs']
    a_vals = sweep_results[cap_val]['a_values']
    min_idx = np.argmin(costs)
    print(f"\n   K={cap_val}: optimal a={a_vals[min_idx]:.2f}, "
          f"cost=${costs[min_idx]:.1f}")

print(f"\n   观察:")
print(f"   - K 越大, 过度保守的负面影响越小 (修得过来)")
print(f"   - K 越小, 最优 a 更接近 0 (不能太保守, 否则排队堵塞)")

print("\n✓ Day 11-13 完成!")
print("  图表已保存: figures/day12_cost_smile_curve.png")
print("  图表已保存: figures/day11_sweep_details.png")
print("  图表已保存: figures/day13_bottleneck_analysis.png")
print("  结果已保存: checkpoints/sweep_results.npz")
