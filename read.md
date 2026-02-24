> **提示：运行代码前请确保安装以下依赖：**
> ```
> pip install torch numpy matplotlib scipy tqdm
> ``` > pip install torch numpy matplotlib scipy tqdm
> ``` 


全部代码已写完。以下是项目结构和运行说明：


## 项目结构

`````
code/
├── requirements.txt              # 依赖库
├── run_all.py                    # 一键运行全部实验
├── src/
│   ├── __init__.py
│   ├── linex_loss.py             # LinEx + PHM08 损失函数
│   ├── data_generator.py         # 异方差退化数据生成器
│   ├── model.py                  # Bi-LSTM + 3FC 网络架构
│   ├── train.py                  # 训练/评估/MC Dropout
│   ├── simulator.py              # FSM 模拟器 (成本/调度/仿真)
│   └── analysis.py               # 全部可视化函数
├── day01_linex_visualization.py  # Day 1: LinEx 损失可视化
├── day02_generate_data.py        # Day 2: 退化数据生成
├── day03_train_mse.py            # Day 3: MSE baseline 训练
├── day04_train_linex.py          # Day 4: LinEx 模型训练
├── day05_variance_sensitivity.py # Day 5: 方差敏感性验证
├── day06_08_build_simulator.py   # Day 6-8: 模拟器构建
├── day09_10_run_simulations.py   # Day 9-10: MSE vs LinEx 模拟
├── day11_13_parameter_analysis.py# Day 11-13: 参数扫描+瓶颈
└── day14_full_report.py          # Day 14: 最终报告
`````







## 运行方式

**⚠️ 需要先安装依赖：**
`````
cd code
pip install torch numpy matplotlib scipy tqdm
`````



**逐天运行（推荐）：**

`````
python day01_linex_visualization.py
python day02_generate_data.py
python day03_train_mse.py
python day04_train_linex.py
python day05_variance_sensitivity.py
# ... 以此类推，按顺序
`````

**一键全跑：**
`````
python run_all.py
`````

## 核心模块说明

| 模块 | 关键内容 |
|---|---|
| `linex_loss.py` | $L(\Delta) = b(e^{a\Delta} - a\Delta - 1)$，带 `torch.clamp` 防溢出 |
| `data_generator.py` | 维纳过程 + 异方差噪声 $\sigma(t) = \sigma_0(1+kt)$ |
| `model.py` | Bi-LSTM(hidden=64) → FC(128) → FC(64) → FC(1)，含 MC Dropout |
| `simulator.py` | 20台机器, 容量K=2, C_failure=100 >> C_preventive=10 |
| `analysis.py` | 全部图表：LinEx曲线、退化数据、预测对比、U型成本曲线等 |

Day 11-13 的参数扫描会为每个 $a$ 值重新训练模型并跑模拟，耗时较长（CPU 上可能需要 30-60 分钟），如果想加速可以减少 `N_EPOCHS_SWEEP` 和 `N_DAYS_SIM`。
