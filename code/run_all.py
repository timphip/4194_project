"""
Master Runner: 一键运行全部 14 天实验
=======================================
按顺序依次执行 Day 1 → Day 14 的所有脚本。
如果某天的脚本失败，会跳过继续运行。

运行方式: python run_all.py
"""

import subprocess
import sys
import os
import time

SCRIPTS = [
    ("Day  1: LinEx Loss Visualization",     "day01_linex_visualization.py"),
    ("Day  2: Degradation Data",             "day02_generate_data.py"),
    ("Day  3: Train MSE Baseline",           "day03_train_mse.py"),
    ("Day  4: Train LinEx Model",            "day04_train_linex.py"),
    ("Day  5: Variance Sensitivity",         "day05_variance_sensitivity.py"),
    ("Day 6-8: Build Simulator",             "day06_08_build_simulator.py"),
    ("Day 9-10: Run Simulations",            "day09_10_run_simulations.py"),
    ("Day 11-13: Parameter Analysis",        "day11_13_parameter_analysis.py"),
    ("Day 14: Full Report",                  "day14_full_report.py"),
]


def main():
    print("=" * 60)
    print("MASTER RUNNER: Risk-Aware Predictive Maintenance")
    print("14-Day Research Plan — Full Execution")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Create output directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    results = []
    total_start = time.time()
    
    for i, (name, script) in enumerate(SCRIPTS):
        print(f"\n{'#' * 60}")
        print(f"# [{i+1}/{len(SCRIPTS)}] {name}")
        print(f"# Script: {script}")
        print(f"{'#' * 60}\n")
        
        start = time.time()
        
        try:
            result = subprocess.run(
                [sys.executable, script],
                cwd=script_dir,
                timeout=600,  # 10 min timeout per script
            )
            elapsed = time.time() - start
            
            if result.returncode == 0:
                status = "✓ SUCCESS"
            else:
                status = f"✗ FAILED (code {result.returncode})"
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            status = "✗ TIMEOUT (>10min)"
        except Exception as e:
            elapsed = time.time() - start
            status = f"✗ ERROR: {e}"
        
        results.append((name, status, elapsed))
        print(f"\n--- {name}: {status} ({elapsed:.1f}s) ---")
    
    # Summary
    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"EXECUTION SUMMARY")
    print(f"{'=' * 60}")
    
    for name, status, elapsed in results:
        print(f"  {status:<20} {elapsed:>6.1f}s  {name}")
    
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    n_success = sum(1 for _, s, _ in results if s.startswith("✓"))
    n_total = len(results)
    print(f"  Result: {n_success}/{n_total} scripts succeeded")
    
    if n_success == n_total:
        print(f"\n  ✓ All experiments completed successfully!")
        print(f"  Check figures/ for all plots and checkpoints/ for model weights.")
    else:
        print(f"\n  Some scripts failed. Check output above for details.")


if __name__ == '__main__':
    main()
