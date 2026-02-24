"""
Day 6-8: 构建 Fleet Selective Maintenance (FSM) 模拟器
========================================================
Day 6: 定义成本与约束
Day 7: 编写贪心调度策略
Day 8: 编写滚动时域模拟器

验收标准：打印日志，能看到机器"运行→维修→重置"的循环。

运行方式: python day06_08_build_simulator.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from src.simulator import CostConfig, calculate_cost, greedy_scheduler, Machine, FleetSimulator

print("=" * 60)
print("Day 6-8: Fleet Selective Maintenance Simulator")
print("=" * 60)

# ============================================================
# Day 6: Cost Definitions
# ============================================================
print("\n" + "=" * 40)
print("Day 6: 成本与约束定义")
print("=" * 40)

config = CostConfig(
    c_preventive=10.0,     # 预防性维修成本
    c_failure=100.0,       # 故障维修成本 (10x preventive!)
    c_waste=1.0,           # 每浪费1单位寿命的成本
    capacity=2,            # 每天最多修 2 台
    n_machines=20,         # 机队 20 台机器
    safety_threshold=15,   # RUL < 15 时考虑维修
)

print(f"\n   成本配置:")
print(f"   {config}")
print(f"\n   决策逻辑:")
print(f"   - 预防性维修: ${config.c_preventive} + ${config.c_waste}/浪费寿命单位")
print(f"   - 意外故障:   ${config.c_failure} (10倍!)")
print(f"   - 每天最多修 {config.capacity} 台 (资源约束!)")
print(f"   - 当预测 RUL < {config.safety_threshold} 时触发维修调度")

# ---- Cost examples ----
print("\n   成本计算示例:")
for action, rul in [('maintain', 30), ('maintain', 5), ('failure', 0)]:
    cost = calculate_cost(action, rul, config)
    print(f"   action={action:<10} true_rul={rul:>3} → cost=${cost:.1f}")

# ============================================================
# Day 7: Greedy Scheduler
# ============================================================
print("\n" + "=" * 40)
print("Day 7: 贪心调度策略")
print("=" * 40)

# Test the scheduler
test_ruls = np.array([50, 8, 120, 3, 45, 12, 200, 7, 90, 30,
                       14, 60, 5, 80, 25, 10, 150, 6, 70, 40])

decisions = greedy_scheduler(test_ruls, capacity=2, safety_threshold=15)

print(f"\n   输入预测 RUL (20 台机器):")
print(f"   {test_ruls}")
print(f"\n   调度决策 (1=维修, 0=不修):")
print(f"   {decisions}")
print(f"\n   被选中维修的机器:")
for i in range(len(decisions)):
    if decisions[i] == 1:
        print(f"   → 机器 {i}: 预测 RUL = {test_ruls[i]} (紧急!)")

print(f"\n   本轮维修 {decisions.sum()} 台 (容量限制: {config.capacity})")

# ============================================================
# Day 8: Rolling Horizon Simulator (Demo Run)
# ============================================================
print("\n" + "=" * 40)
print("Day 8: 滚动时域模拟器 (Oracle 测试)")
print("=" * 40)

print("\n   使用 Oracle (真实 RUL) 运行 100 天模拟...")
print("   (验证模拟器逻辑正确)")

# Oracle simulator (no model, uses true RUL)
sim = FleetSimulator(
    config=config,
    model=None,  # Oracle: uses true RUL
    rul_max=1.0,
    window_size=30,
    seed=42
)

results = sim.run(n_days=100, verbose=True)

print(f"\n   --- Oracle 100天模拟结果 ---")
print(f"   总成本: ${results['total_cost']:.1f}")
print(f"   总故障: {results['total_failures']} 次")
print(f"   总维修: {results['total_maintenances']} 次")
print(f"   日均成本: ${results['avg_daily_cost']:.2f}")
print(f"   日均故障率: {results['failure_rate']:.3f}")

# Show some machines current state
print(f"\n   当前机器状态 (前5台):")
for m in sim.machines[:5]:
    print(f"   Machine {m.machine_id}: step={m.current_step}, "
          f"health={m.current_health:.1f}, "
          f"true_rul={m.true_rul:.0f}, "
          f"status={m.status}")

print("\n✓ Day 6-8 完成!")
print("  模拟器已通过 Oracle 测试")
print("  下一步: 用训练好的 MSE/LinEx 模型接入模拟器")
