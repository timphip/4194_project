"""
Day 5 & Day 11-13: Analysis & Visualization Utilities
=======================================================
Functions for plotting, variance analysis, parameter sweeping,
and generating publication-quality figures.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---- Global style ----
rcParams['figure.figsize'] = (10, 6)
rcParams['font.size'] = 12
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.3
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')


def ensure_figure_dir():
    os.makedirs(FIGURE_DIR, exist_ok=True)


# ==================================================================
# Day 1: LinEx Loss Visualization
# ==================================================================

def plot_linex_curves(save=True):
    """Plot LinEx loss vs MSE for different values of shape parameter a."""
    from src.linex_loss import linex_loss_numpy, mse_loss_numpy
    
    errors = np.linspace(-5, 5, 500)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: different a values
    ax = axes[0]
    for a_val in [0.1, 0.3, 0.5, 1.0, 2.0]:
        loss = linex_loss_numpy(errors, a=a_val)
        ax.plot(errors, loss, label=f'LinEx (a={a_val})')
    
    ax.plot(errors, mse_loss_numpy(errors), 'k--', label='MSE', linewidth=2)
    ax.set_xlabel('Prediction Error (Δ = ŷ - y)')
    ax.set_ylabel('Loss')
    ax.set_title('LinEx Loss: Varying Shape Parameter a')
    ax.legend()
    ax.set_ylim(-1, 30)
    
    # Right: Zoom on a=0.5 vs MSE
    ax = axes[1]
    a_val = 0.5
    loss_linex = linex_loss_numpy(errors, a=a_val)
    loss_mse = mse_loss_numpy(errors)
    
    ax.fill_between(errors[errors > 0], 0, loss_linex[errors > 0],
                     alpha=0.3, color='red', label='Overestimation (DANGER)')
    ax.fill_between(errors[errors < 0], 0, loss_linex[errors < 0],
                     alpha=0.3, color='green', label='Underestimation (safe)')
    ax.plot(errors, loss_linex, 'r-', linewidth=2, label=f'LinEx (a={a_val})')
    ax.plot(errors, loss_mse, 'k--', linewidth=2, label='MSE')
    ax.set_xlabel('Prediction Error (Δ = ŷ - y)')
    ax.set_ylabel('Loss')
    ax.set_title('Asymmetric Risk: LinEx vs MSE')
    ax.legend()
    ax.set_ylim(-1, 20)
    
    plt.tight_layout()
    
    if save:
        ensure_figure_dir()
        plt.savefig(os.path.join(FIGURE_DIR, 'day01_linex_loss.png'), dpi=150)
    plt.show()


# ==================================================================
# Day 2: Degradation Curves Visualization
# ==================================================================

def plot_degradation_curves(all_health, n_curves=5, save=True):
    """Plot sample degradation curves showing heteroscedastic behavior."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: sample curves
    ax = axes[0]
    for i in range(min(n_curves, len(all_health))):
        t = np.arange(len(all_health[i]))
        ax.plot(t, all_health[i], alpha=0.8, label=f'Machine {i+1}')
    ax.axhline(y=0, color='r', linestyle='--', label='Failure Threshold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Health Index')
    ax.set_title('Sample Degradation Curves (Random Walk + Heteroscedastic Noise)')
    ax.legend()
    
    # Right: rolling std to show heteroscedasticity
    ax = axes[1]
    for i in range(min(n_curves, len(all_health))):
        h = all_health[i]
        # Compute diff (step-to-step changes)
        diffs = np.diff(h)
        # Rolling window std
        w = 20
        if len(diffs) > w:
            rolling_std = np.array([
                np.std(diffs[max(0, j-w):j+1])
                for j in range(len(diffs))
            ])
            ax.plot(rolling_std, alpha=0.8, label=f'Machine {i+1}')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Rolling Std of Health Changes (window=20)')
    ax.set_title('Heteroscedasticity: Noise Variance Increases with Age')
    ax.legend()
    
    plt.tight_layout()
    
    if save:
        ensure_figure_dir()
        plt.savefig(os.path.join(FIGURE_DIR, 'day02_degradation_curves.png'), dpi=150)
    plt.show()


# ==================================================================
# Day 3-4: Prediction Comparison Plot
# ==================================================================

def plot_prediction_comparison(
    preds_mse, preds_linex, trues,
    n_points=500, save=True
):
    """
    Plot MSE vs LinEx predictions against true RUL.
    Shows that LinEx predictions are biased downward (conservative).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Select subset for clarity
    idx = np.random.choice(len(trues), min(n_points, len(trues)), replace=False)
    idx = np.sort(idx)
    
    trues_sub = trues[idx]
    mse_sub = preds_mse[idx]
    linex_sub = preds_linex[idx]
    
    # Sort by true RUL for line plot
    sort_idx = np.argsort(trues_sub)
    trues_sorted = trues_sub[sort_idx]
    mse_sorted = mse_sub[sort_idx]
    linex_sorted = linex_sub[sort_idx]
    
    # Left: Line plot
    ax = axes[0]
    ax.plot(trues_sorted, trues_sorted, 'k--', linewidth=2, label='Perfect (y=x)')
    ax.plot(trues_sorted, mse_sorted, 'b.', alpha=0.3, markersize=3, label='MSE Model')
    ax.plot(trues_sorted, linex_sorted, 'r.', alpha=0.3, markersize=3, label='LinEx Model')
    ax.set_xlabel('True RUL')
    ax.set_ylabel('Predicted RUL')
    ax.set_title('Predicted vs True RUL')
    ax.legend()
    
    # Right: Bias histogram
    ax = axes[1]
    bias_mse = preds_mse - trues
    bias_linex = preds_linex - trues
    
    ax.hist(bias_mse, bins=50, alpha=0.5, color='blue', label=f'MSE (mean bias={np.mean(bias_mse):.1f})')
    ax.hist(bias_linex, bins=50, alpha=0.5, color='red', label=f'LinEx (mean bias={np.mean(bias_linex):.1f})')
    ax.axvline(x=0, color='k', linestyle='--')
    ax.set_xlabel('Prediction Bias (Pred - True)')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Bias Distribution')
    ax.legend()
    
    plt.tight_layout()
    
    if save:
        ensure_figure_dir()
        plt.savefig(os.path.join(FIGURE_DIR, 'day04_prediction_comparison.png'), dpi=150)
    plt.show()
    
    # Print statistics
    print(f"\n--- Prediction Statistics ---")
    print(f"MSE Model:   RMSE={np.sqrt(np.mean(bias_mse**2)):.2f}, "
          f"Mean Bias={np.mean(bias_mse):.2f}, Median Bias={np.median(bias_mse):.2f}")
    print(f"LinEx Model: RMSE={np.sqrt(np.mean(bias_linex**2)):.2f}, "
          f"Mean Bias={np.mean(bias_linex):.2f}, Median Bias={np.median(bias_linex):.2f}")


# ==================================================================
# Day 5: Variance-Bias Sensitivity Analysis
# ==================================================================

def plot_variance_sensitivity(preds, trues, test_health, window_size=30, save=True):
    """
    Scatter plot: X = local variance, Y = prediction bias.
    Theory: higher variance → more negative bias (more conservative).
    """
    # Compute sliding window variance for each test sample
    variances = []
    sample_idx = 0
    
    for health in test_health:
        health_norm = health / 100.0
        for i in range(len(health) - window_size):
            window = health_norm[i:i + window_size]
            var = np.var(window)
            variances.append(var)
    
    variances = np.array(variances[:len(preds)])
    bias = preds - trues
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Scatter
    ax = axes[0]
    ax.scatter(variances, bias, alpha=0.2, s=5, c='steelblue')
    
    # Add trend line
    if len(variances) > 10:
        z = np.polyfit(variances, bias, 1)
        p = np.poly1d(z)
        var_range = np.linspace(variances.min(), variances.max(), 100)
        ax.plot(var_range, p(var_range), 'r-', linewidth=2,
                label=f'Trend: slope={z[0]:.1f}')
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Local Variance (sliding window)')
    ax.set_ylabel('Prediction Bias (Pred - True)')
    ax.set_title('Variance Sensitivity: Uncertainty → Conservative Bias')
    ax.legend()
    
    # Right: Binned average
    ax = axes[1]
    n_bins = 20
    var_bins = np.linspace(variances.min(), variances.max(), n_bins + 1)
    bin_centers = []
    bin_means = []
    bin_stds = []
    
    for j in range(n_bins):
        mask = (variances >= var_bins[j]) & (variances < var_bins[j + 1])
        if mask.sum() > 5:
            bin_centers.append((var_bins[j] + var_bins[j + 1]) / 2)
            bin_means.append(np.mean(bias[mask]))
            bin_stds.append(np.std(bias[mask]) / np.sqrt(mask.sum()))
    
    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds,
                fmt='o-', color='steelblue', capsize=3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Local Variance (binned)')
    ax.set_ylabel('Mean Prediction Bias')
    ax.set_title('Binned Variance vs Mean Bias')
    
    plt.tight_layout()
    
    if save:
        ensure_figure_dir()
        plt.savefig(os.path.join(FIGURE_DIR, 'day05_variance_sensitivity.png'), dpi=150)
    plt.show()


# ==================================================================
# Day 9-10: Simulation Results Comparison
# ==================================================================

def plot_simulation_comparison(results_mse, results_linex, save=True):
    """Compare MSE vs LinEx model simulation outcomes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: cumulative cost
    ax = axes[0, 0]
    cum_mse = np.cumsum(results_mse['daily_costs'])
    cum_linex = np.cumsum(results_linex['daily_costs'])
    ax.plot(cum_mse, 'b-', label='MSE Model', alpha=0.8)
    ax.plot(cum_linex, 'r-', label='LinEx Model', alpha=0.8)
    ax.set_xlabel('Day')
    ax.set_ylabel('Cumulative Cost')
    ax.set_title('Cumulative Fleet Maintenance Cost')
    ax.legend()
    
    # Top-right: daily cost rolling average
    ax = axes[0, 1]
    w = 50
    if len(results_mse['daily_costs']) > w:
        roll_mse = np.convolve(results_mse['daily_costs'], np.ones(w)/w, mode='valid')
        roll_linex = np.convolve(results_linex['daily_costs'], np.ones(w)/w, mode='valid')
        ax.plot(roll_mse, 'b-', label='MSE Model', alpha=0.8)
        ax.plot(roll_linex, 'r-', label='LinEx Model', alpha=0.8)
    ax.set_xlabel('Day')
    ax.set_ylabel(f'Rolling Avg Cost (window={w})')
    ax.set_title('Daily Cost Trend')
    ax.legend()
    
    # Bottom-left: bar chart comparison
    ax = axes[1, 0]
    metrics = ['Total Cost', 'Failures', 'Maintenances']
    mse_vals = [results_mse['total_cost'], results_mse['total_failures'],
                results_mse['total_maintenances']]
    linex_vals = [results_linex['total_cost'], results_linex['total_failures'],
                  results_linex['total_maintenances']]
    
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax.bar(x - width/2, mse_vals, width, label='MSE', color='steelblue')
    bars2 = ax.bar(x + width/2, linex_vals, width, label='LinEx', color='indianred')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title('Simulation Summary')
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9)
    
    # Bottom-right: text summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = (
        f"{'Metric':<25} {'MSE':>12} {'LinEx':>12} {'Δ':>12}\n"
        f"{'='*60}\n"
        f"{'Total Cost':<25} {results_mse['total_cost']:>12.1f} "
        f"{results_linex['total_cost']:>12.1f} "
        f"{results_linex['total_cost'] - results_mse['total_cost']:>+12.1f}\n"
        f"{'Total Failures':<25} {results_mse['total_failures']:>12d} "
        f"{results_linex['total_failures']:>12d} "
        f"{results_linex['total_failures'] - results_mse['total_failures']:>+12d}\n"
        f"{'Total Maintenances':<25} {results_mse['total_maintenances']:>12d} "
        f"{results_linex['total_maintenances']:>12d} "
        f"{results_linex['total_maintenances'] - results_mse['total_maintenances']:>+12d}\n"
        f"{'Avg Daily Cost':<25} {results_mse['avg_daily_cost']:>12.2f} "
        f"{results_linex['avg_daily_cost']:>12.2f} "
        f"{results_linex['avg_daily_cost'] - results_mse['avg_daily_cost']:>+12.2f}\n"
        f"{'Failure Rate/Day':<25} {results_mse['failure_rate']:>12.3f} "
        f"{results_linex['failure_rate']:>12.3f} "
        f"{results_linex['failure_rate'] - results_mse['failure_rate']:>+12.3f}\n"
    )
    ax.text(0.1, 0.5, summary, transform=ax.transAxes,
            fontsize=10, family='monospace', verticalalignment='center')
    ax.set_title('Comparison Summary')
    
    plt.tight_layout()
    
    if save:
        ensure_figure_dir()
        plt.savefig(os.path.join(FIGURE_DIR, 'day09_10_simulation_comparison.png'), dpi=150)
    plt.show()


# ==================================================================
# Day 11-12: Parameter Sweep & Cost Smile Curve
# ==================================================================

def plot_cost_smile_curve(a_values, costs, save=True):
    """
    Plot the "Cost Smile Curve": total cost vs LinEx parameter a.
    Should show a U-shape:
        - a≈0 (MSE): many failures → expensive
        - a too large: over-conservative → resource congestion → expensive
        - Sweet spot in the middle
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(a_values, costs, 'o-', color='steelblue', linewidth=2, markersize=6)
    
    # Annotate minimum
    min_idx = np.argmin(costs)
    ax.annotate(
        f'Optimal: a={a_values[min_idx]:.2f}\nCost={costs[min_idx]:.0f}',
        xy=(a_values[min_idx], costs[min_idx]),
        xytext=(a_values[min_idx] + 0.3, costs[min_idx] + (max(costs) - min(costs)) * 0.15),
        arrowprops=dict(arrowstyle='->', color='red'),
        fontsize=12, color='red', fontweight='bold'
    )
    
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='a=0 (≈MSE)')
    
    ax.set_xlabel('LinEx Shape Parameter (a)', fontsize=14)
    ax.set_ylabel('Total Fleet Maintenance Cost', fontsize=14)
    ax.set_title('Cost Smile Curve: Finding the Golden a', fontsize=16)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    if save:
        ensure_figure_dir()
        plt.savefig(os.path.join(FIGURE_DIR, 'day12_cost_smile_curve.png'), dpi=150)
    plt.show()
    
    print(f"\n--- Parameter Sweep Results ---")
    print(f"Optimal a = {a_values[min_idx]:.2f}")
    print(f"Minimum Cost = {costs[min_idx]:.1f}")
    print(f"MSE Cost (a≈0) ≈ {costs[0]:.1f}")
    print(f"Cost Reduction: {(1 - costs[min_idx]/costs[0])*100:.1f}%")


# ==================================================================
# Day 13: Bottleneck (Capacity) Analysis
# ==================================================================

def plot_capacity_analysis(capacity_values, sweep_results, save=True):
    """
    Plot cost smile curves for different maintenance capacities.
    More capacity → less impact from over-conservatism.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: overlay curves
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(capacity_values)))
    
    for cap_val, color in zip(capacity_values, colors):
        a_values = sweep_results[cap_val]['a_values']
        costs = sweep_results[cap_val]['costs']
        ax.plot(a_values, costs, 'o-', color=color, label=f'K={cap_val}', linewidth=2)
    
    ax.set_xlabel('LinEx Shape Parameter (a)')
    ax.set_ylabel('Total Fleet Maintenance Cost')
    ax.set_title('Cost Curves Under Different Capacity Constraints')
    ax.legend()
    
    # Right: optimal a vs capacity
    ax = axes[1]
    optimal_a = []
    min_costs = []
    for cap_val in capacity_values:
        costs = sweep_results[cap_val]['costs']
        a_vals = sweep_results[cap_val]['a_values']
        min_idx = np.argmin(costs)
        optimal_a.append(a_vals[min_idx])
        min_costs.append(costs[min_idx])
    
    ax2 = ax.twinx()
    
    l1 = ax.bar(np.arange(len(capacity_values)) - 0.15, optimal_a, 0.3,
                color='steelblue', alpha=0.8, label='Optimal a')
    l2 = ax2.bar(np.arange(len(capacity_values)) + 0.15, min_costs, 0.3,
                 color='indianred', alpha=0.8, label='Min Cost')
    
    ax.set_xticks(np.arange(len(capacity_values)))
    ax.set_xticklabels([f'K={k}' for k in capacity_values])
    ax.set_xlabel('Maintenance Capacity')
    ax.set_ylabel('Optimal a', color='steelblue')
    ax2.set_ylabel('Minimum Cost', color='indianred')
    ax.set_title('Bottleneck Analysis: Capacity vs Optimal Risk Aversion')
    
    lines = [l1, l2]
    labels = ['Optimal a', 'Min Cost']
    ax.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    
    if save:
        ensure_figure_dir()
        plt.savefig(os.path.join(FIGURE_DIR, 'day13_bottleneck_analysis.png'), dpi=150)
    plt.show()


# ==================================================================
# Day 14: Final Three-Panel Report Figure
# ==================================================================

def plot_final_report(all_health, preds_mse, preds_linex, trues,
                      a_values, costs, save=True):
    """
    Three-panel figure for the final report:
        Panel 1: Degradation curves (Day 2)
        Panel 2: Prediction comparison (Day 4)
        Panel 3: Cost smile curve (Day 12)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Degradation Curves
    ax = axes[0]
    for i in range(min(5, len(all_health))):
        t = np.arange(len(all_health[i]))
        ax.plot(t, all_health[i], alpha=0.7, label=f'Machine {i+1}')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Health Index')
    ax.set_title('(a) Degradation Curves\n(Heteroscedastic Noise)')
    ax.legend(fontsize=8)
    
    # Panel 2: Prediction Comparison
    ax = axes[1]
    sort_idx = np.argsort(trues)
    n_show = min(500, len(trues))
    step = max(1, len(trues) // n_show)
    show_idx = sort_idx[::step]
    
    ax.plot(trues[show_idx], trues[show_idx], 'k--', linewidth=2, label='Perfect')
    ax.scatter(trues[show_idx], preds_mse[show_idx], alpha=0.3, s=5,
               c='blue', label='MSE')
    ax.scatter(trues[show_idx], preds_linex[show_idx], alpha=0.3, s=5,
               c='red', label='LinEx')
    ax.set_xlabel('True RUL')
    ax.set_ylabel('Predicted RUL')
    ax.set_title('(b) Prediction: MSE vs LinEx\n(LinEx biased downward = conservative)')
    ax.legend(fontsize=8)
    
    # Panel 3: Cost Smile Curve
    ax = axes[2]
    ax.plot(a_values, costs, 'o-', color='steelblue', linewidth=2)
    min_idx = np.argmin(costs)
    ax.plot(a_values[min_idx], costs[min_idx], 'r*', markersize=15,
            label=f'Optimal a={a_values[min_idx]:.2f}')
    ax.set_xlabel('LinEx Parameter a')
    ax.set_ylabel('Total Fleet Cost')
    ax.set_title('(c) Cost Smile Curve\n(U-shape trade-off)')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    
    if save:
        ensure_figure_dir()
        plt.savefig(os.path.join(FIGURE_DIR, 'day14_final_report.png'), dpi=200)
    plt.show()
    
    print("\n✓ Final report figure saved!")
