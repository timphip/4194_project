"""
Day 2: Synthetic Degradation Data Generator
=============================================
Generates heteroscedastic degradation curves using random walk (Wiener process).

Physical model:
    health[t] = health[t-1] - λ + ε(t)
    ε(t) ~ N(0, σ(t)²)
    σ(t) = σ_base * (1 + k * t)   ← heteroscedastic noise

Key features:
    - Random walk with drift (simulates wear-out degradation)
    - Heteroscedastic noise (variance grows with age — realistic!)
    - Fleet heterogeneity (each machine has slightly different parameters)
    - Ground-truth RUL available for every time step
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ==================================================================
# Single Machine Degradation Curve
# ==================================================================

def generate_single_curve(
    max_life=300,
    degradation_rate=0.5,
    noise_base=0.3,
    noise_growth=0.005,
    initial_health=100.0,
    failure_threshold=0.0,
    seed=None
):
    """
    Generate one degradation curve with heteroscedastic noise.
    
    Args:
        max_life: maximum possible time steps
        degradation_rate: λ, average health loss per step
        noise_base: σ_base, base noise standard deviation
        noise_growth: k, rate of noise variance increase
        initial_health: starting health index
        failure_threshold: health level at which machine fails
        seed: random seed for reproducibility
    
    Returns:
        health: np.array of health values [T+1]
        rul:    np.array of true RUL values [T+1]
        failure_time: int, time step of failure
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    health_list = [initial_health]
    t = 0
    
    while t < max_life:
        t += 1
        # Heteroscedastic noise: σ(t) = σ_base * (1 + k * t)
        noise_std = noise_base * (1.0 + noise_growth * t)
        noise = rng.normal(0, noise_std)
        
        new_health = health_list[-1] - degradation_rate + noise
        
        if new_health <= failure_threshold:
            health_list.append(failure_threshold)
            break
        
        health_list.append(new_health)
    
    health = np.array(health_list)
    failure_time = len(health) - 1
    
    # True RUL at each time step
    rul = np.array([failure_time - i for i in range(len(health))], dtype=np.float32)
    rul = np.maximum(rul, 0)
    
    return health, rul, failure_time


# ==================================================================
# Fleet Data Generation
# ==================================================================

def generate_fleet_data(n_machines=100, seed=42):
    """
    Generate degradation data for a fleet of machines.
    Each machine has slightly different degradation parameters
    (fleet heterogeneity).
    
    Args:
        n_machines: number of machines in the fleet
        seed: master random seed
    
    Returns:
        all_health: list of health arrays
        all_rul:    list of RUL arrays
        all_failure_times: list of failure time ints
    """
    rng = np.random.RandomState(seed)
    
    all_health = []
    all_rul = []
    all_failure_times = []
    
    for i in range(n_machines):
        # Fleet heterogeneity: slightly different parameters per machine
        deg_rate = rng.uniform(0.3, 0.7)
        noise_base = rng.uniform(0.2, 0.5)
        noise_growth = rng.uniform(0.003, 0.008)
        
        health, rul, ft = generate_single_curve(
            max_life=300,
            degradation_rate=deg_rate,
            noise_base=noise_base,
            noise_growth=noise_growth,
            initial_health=100.0,
            failure_threshold=0.0,
            seed=seed * 100 + i
        )
        
        all_health.append(health)
        all_rul.append(rul)
        all_failure_times.append(ft)
    
    return all_health, all_rul, all_failure_times


# ==================================================================
# PyTorch Dataset (Sliding Window)
# ==================================================================

class RULDataset(Dataset):
    """
    Sliding window dataset for RUL prediction.
    
    Each sample: 
        X = [health[t-W+1], ..., health[t]]  → window of W readings
        Y = RUL[t]                            → remaining useful life at time t
    
    Health is normalized to [0, 1] by dividing by initial_health (100).
    RUL is normalized to [0, 1] by dividing by rul_max.
    """
    
    def __init__(self, health_curves, rul_curves, window_size=30):
        self.window_size = window_size
        samples_X = []
        samples_Y = []
        
        for health, rul in zip(health_curves, rul_curves):
            # Normalize health to [0, 1]
            health_norm = health / 100.0
            
            # Create sliding windows
            for i in range(len(health) - window_size):
                x = health_norm[i:i + window_size]
                y = rul[i + window_size - 1]  # RUL at end of window
                samples_X.append(x)
                samples_Y.append(y)
        
        self.samples_X = np.array(samples_X, dtype=np.float32)
        self.samples_Y = np.array(samples_Y, dtype=np.float32)
        
        # Compute normalization factor for RUL
        self.rul_max = float(self.samples_Y.max()) if len(self.samples_Y) > 0 else 1.0
        self.samples_Y_norm = self.samples_Y / self.rul_max
    
    def __len__(self):
        return len(self.samples_X)
    
    def __getitem__(self, idx):
        # x shape: [window_size, 1] for LSTM input
        x = torch.FloatTensor(self.samples_X[idx]).unsqueeze(-1)
        y = torch.FloatTensor([self.samples_Y_norm[idx]])
        return x, y


# ==================================================================
# DataLoader Factory
# ==================================================================

def create_dataloaders(
    n_machines=100,
    window_size=30,
    batch_size=64,
    train_ratio=0.8,
    seed=42
):
    """
    Create train/test dataloaders with consistent normalization.
    
    Returns:
        train_loader: DataLoader for training
        test_loader:  DataLoader for testing
        rul_max:      float, RUL normalization factor (to denormalize predictions)
        test_health:  list of test health curves (for analysis)
        test_rul:     list of test RUL curves (for analysis)
    """
    all_health, all_rul, _ = generate_fleet_data(n_machines, seed)
    
    n_train = int(n_machines * train_ratio)
    
    train_health = all_health[:n_train]
    train_rul = all_rul[:n_train]
    test_health = all_health[n_train:]
    test_rul = all_rul[n_train:]
    
    train_dataset = RULDataset(train_health, train_rul, window_size)
    test_dataset = RULDataset(test_health, test_rul, window_size)
    
    # Use training set's rul_max for normalization consistency
    rul_max = train_dataset.rul_max
    test_dataset.rul_max = rul_max
    test_dataset.samples_Y_norm = test_dataset.samples_Y / rul_max
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    
    return train_loader, test_loader, rul_max, test_health, test_rul
