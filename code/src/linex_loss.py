"""
Day 1: LinEx Loss Function Implementation
==========================================
Linear-Exponential (LinEx) Loss for asymmetric risk-aware prediction.

Formula:
    L(Δ) = b * (exp(a*Δ) - a*Δ - 1)
    where Δ = y_pred - y_true

Properties:
    - a > 0: severely penalizes overestimation (predicting RUL too high → danger)
    - a < 0: severely penalizes underestimation (predicting RUL too low → waste)
    - a → 0: converges to 0.5 * b * a² * Δ² ≈ MSE (symmetric, risk-neutral)
    - Strictly convex: d²L/dΔ² = b*a²*exp(a*Δ) > 0

Bayesian Optimal Predictor (Gaussian posterior):
    θ̂_Bayes = μ - a*σ²/2
    → Higher uncertainty (σ²) → larger conservative shift
    → Higher risk aversion (a) → larger conservative shift

Reference: Varian (1975), Zellner (1986)
"""

import numpy as np
import torch
import torch.nn as nn


# ==================================================================
# NumPy version (for visualization and analysis)
# ==================================================================

def linex_loss_numpy(error, a, b=1.0):
    """
    Compute LinEx loss for given error values (NumPy).
    
    Args:
        error: Δ = y_pred - y_true (scalar or array)
        a: shape parameter (a>0 penalizes overestimation)
        b: scale parameter (default 1.0)
    
    Returns:
        Loss values (same shape as error)
    """
    return b * (np.exp(a * error) - a * error - 1)


def mse_loss_numpy(error):
    """Standard MSE for comparison."""
    return error ** 2


def mae_loss_numpy(error):
    """Standard MAE for comparison."""
    return np.abs(error)


# ==================================================================
# PyTorch version (for model training)
# ==================================================================

class LinExLoss(nn.Module):
    """
    PyTorch LinEx loss with numerical stability.
    
    Uses torch.clamp on a*Δ to prevent exp() overflow,
    and torch.expm1 for stability near zero.
    
    Args:
        a (float): shape parameter. a>0 penalizes overestimation (high RUL).
        b (float): scale parameter. Default 1.0.
    """
    
    def __init__(self, a=0.1, b=1.0):
        super().__init__()
        self.a = a
        self.b = b
    
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: predicted values [batch_size, 1] or [batch_size]
            y_true: ground truth values [batch_size, 1] or [batch_size]
        
        Returns:
            Scalar mean loss
        """
        delta = y_pred.view(-1) - y_true.view(-1)
        
        # Clamp to prevent overflow: exp(50) ≈ 5.18e21, safe for float32
        a_delta = torch.clamp(self.a * delta, max=50.0, min=-50.0)
        
        # expm1(x) = exp(x) - 1, numerically stable near x=0
        # LinEx = b * (exp(a*Δ) - a*Δ - 1) = b * (expm1(a*Δ) - a*Δ)
        loss = self.b * (torch.expm1(a_delta) - self.a * delta)
        
        return torch.mean(loss)


class PHM08ScoreLoss(nn.Module):
    """
    Smooth approximation of the PHM08 scoring function.
    
    Original (non-differentiable at 0):
        s = exp(-d/13) - 1  if d < 0 (underestimation)
        s = exp(d/10) - 1   if d >= 0 (overestimation)
    
    Smooth version using softplus blending.
    """
    
    def __init__(self, a1=13.0, a2=10.0):
        super().__init__()
        self.a1 = a1
        self.a2 = a2
    
    def forward(self, y_pred, y_true):
        d = y_pred.view(-1) - y_true.view(-1)
        
        # Smooth blending using sigmoid
        sigmoid_d = torch.sigmoid(10.0 * d)  # sharp transition near 0
        
        # Underestimation branch: exp(-d/a1) - 1  (d < 0 → -d > 0)
        under = torch.expm1(torch.clamp(-d / self.a1, max=50.0))
        # Overestimation branch: exp(d/a2) - 1    (d > 0)
        over = torch.expm1(torch.clamp(d / self.a2, max=50.0))
        
        score = (1 - sigmoid_d) * under + sigmoid_d * over
        return torch.mean(score)
