"""
Day 14: 整理 Toy Model 最终报告
=================================
将 Day 2（退化数据）、Day 4（预测对比）、Day 12（成本曲线）整合为三联图。
输出完整结论。

运行方式: python day14_full_report.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from src.data_generator import generate_fleet_data, create_dataloaders
from src.model import BiLSTM_RUL
from src.train import get_predictions
from src.analysis import plot_final_report

print("=" * 60)
print("Day 14: Final Report Generation")
print("=" * 60)

# ---- Configuration ----
WINDOW_SIZE = 30
HIDDEN_SIZE = 64
FC_DIM = 128
DROPOUT = 0.3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- 1. Load data ----
print("\n[1] 加载数据...")
all_health, all_rul, all_ft = generate_fleet_data(n_machines=100, seed=42)

_, test_loader, rul_max, test_health, test_rul = create_dataloaders(
    n_machines=100, window_size=WINDOW_SIZE,
    batch_size=64, train_ratio=0.8, seed=42
)

# ---- 2. Load models ----
print("[2] 加载模型...")
model_mse = BiLSTM_RUL(input_size=1, hidden_size=HIDDEN_SIZE,
                        fc_dim=FC_DIM, dropout=DROPOUT)
model_mse.load_state_dict(
    torch.load('checkpoints/model_mse.pt', weights_only=True,
               map_location=DEVICE))
model_mse.to(DEVICE)

model_linex = BiLSTM_RUL(input_size=1, hidden_size=HIDDEN_SIZE,
                           fc_dim=FC_DIM, dropout=DROPOUT)
model_linex.load_state_dict(
    torch.load('checkpoints/model_linex.pt', weights_only=True,
               map_location=DEVICE))
model_linex.to(DEVICE)

# ---- 3. Get predictions ----
print("[3] 获取预测值...")
preds_mse, trues = get_predictions(model_mse, test_loader, rul_max, DEVICE)
preds_linex, _ = get_predictions(model_linex, test_loader, rul_max, DEVICE)

# ---- 4. Load sweep results ----
print("[4] 加载参数扫描结果...")
sweep = np.load('checkpoints/sweep_results.npz')
a_values = sweep['a_values']
costs = sweep['costs']

# ---- 5. Generate final report figure ----
print("[5] 生成最终三联图...")
plot_final_report(
    all_health=all_health,
    preds_mse=preds_mse,
    preds_linex=preds_linex,
    trues=trues,
    a_values=a_values,
    costs=costs,
    save=True
)

# ---- 6. Print final conclusions ----
print("\n" + "=" * 60)
print("FINAL CONCLUSIONS")
print("=" * 60)

bias_mse = np.mean(preds_mse - trues)
bias_linex = np.mean(preds_linex - trues)
rmse_mse = np.sqrt(np.mean((preds_mse - trues)**2))
rmse_linex = np.sqrt(np.mean((preds_linex - trues)**2))
overest_mse = np.mean(preds_mse > trues) * 100
overest_linex = np.mean(preds_linex > trues) * 100
min_idx = np.argmin(costs)

# Load simulation results
sim_data = np.load('checkpoints/simulation_results.npz')

print(f"""
1. 数据特性 (Day 2):
   - 退化数据展现明显的异方差性: 尾部噪声更大
   - 机队异质性: 每台机器退化速率不同
   - 这种特性使得传统 MSE 模型在寿命末期更容易高估 RUL

2. 预测对比 (Day 3-4):
   - MSE 模型:   RMSE={rmse_mse:.1f}, Bias={bias_mse:+.1f}, 高估率={overest_mse:.1f}%
   - LinEx 模型:  RMSE={rmse_linex:.1f}, Bias={bias_linex:+.1f}, 高估率={overest_linex:.1f}%
   - LinEx 成功引入保守偏差, 降低了高估 RUL 的危险

3. 方差敏感性 (Day 5):
   - 验证了理论: 不确定性越大, LinEx 给出越保守的预测
   - θ̂_Bayes = μ - a*σ²/2 在实验中得到定性验证

4. 模拟对比 (Day 9-10):
   - MSE 模拟: 成本=${float(sim_data['mse_cost']):.0f}, \
故障={int(sim_data['mse_failures'])}次
   - LinEx 模拟: 成本=${float(sim_data['linex_cost']):.0f}, \
故障={int(sim_data['linex_failures'])}次

5. 成本微笑曲线 (Day 12):
   - 最优 a = {a_values[min_idx]:.2f}
   - 曲线呈 U 型: a 太小 → 故障多; a 太大 → 资源挤兑
   - 存在一个"黄金平衡点"

6. 瓶颈分析 (Day 13):
   - 资源越充足 (K越大), LinEx 过度保守的负面影响越小
   - 资源受限时, 需要更精细地调节 a 以避免排队堵塞

核心结论:
   决策感知的非对称损失函数 (LinEx) 能够在预测阶段内嵌风险偏好,
   通过牺牲少量预测精度 (RMSE略增), 显著降低了高风险高估率,
   在有限维修资源约束下实现更低的总运营成本。
   这验证了 "统计精度 ≠ 决策最优性" 的核心论断。
""")

print("=" * 60)
print("✓ 项目完成!")
print("=" * 60)
print(f"""
输出文件:
  figures/day01_linex_loss.png           - LinEx 损失函数可视化
  figures/day02_degradation_curves.png   - 退化曲线 (异方差)
  figures/day03_mse_baseline.png         - MSE baseline 结果
  figures/day04_prediction_comparison.png - MSE vs LinEx 预测对比
  figures/day05_variance_sensitivity.png - 方差敏感性分析
  figures/day05_mc_dropout_analysis.png  - MC Dropout 不确定性
  figures/day09_10_simulation_comparison.png - 模拟对比
  figures/day11_sweep_details.png        - 参数扫描详情
  figures/day12_cost_smile_curve.png     - 成本微笑曲线
  figures/day13_bottleneck_analysis.png  - 瓶颈分析
  figures/day14_final_report.png         - 最终三联图

模型与数据:
  checkpoints/model_mse.pt              - MSE 模型权重
  checkpoints/model_linex.pt            - LinEx 模型权重
  checkpoints/mse_predictions.npz       - MSE 预测结果
  checkpoints/linex_predictions.npz     - LinEx 预测结果
  checkpoints/simulation_results.npz    - 模拟结果
  checkpoints/sweep_results.npz         - 参数扫描结果
""")
