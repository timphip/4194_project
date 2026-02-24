"""
Day 3-4: Training Utilities
============================
Functions for training and evaluating Bi-LSTM models with
different loss functions (MSE, LinEx, PHM08).
"""

import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.model import BiLSTM_RUL
from src.linex_loss import LinExLoss


def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    rul_max,
    n_epochs=100,
    lr=1e-3,
    device='cpu',
    save_path=None,
    verbose=True
):
    """
    Train a model with the given loss function.
    
    Args:
        model:        nn.Module (BiLSTM_RUL)
        train_loader: DataLoader for training
        test_loader:  DataLoader for testing
        criterion:    loss function (nn.MSELoss, LinExLoss, etc.)
        rul_max:      RUL normalization factor
        n_epochs:     number of training epochs
        lr:           learning rate
        device:       'cpu' or 'cuda'
        save_path:    path to save best model checkpoint
        verbose:      whether to print progress
    
    Returns:
        train_losses: list of per-epoch training losses
        test_losses:  list of per-epoch test losses (RMSE in original scale)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )
    
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    iterator = tqdm(range(n_epochs), desc="Training") if verbose else range(n_epochs)
    
    for epoch in iterator:
        # ---- Training ----
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_Y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)
        
        # ---- Evaluation (RMSE in original RUL scale) ----
        test_rmse = evaluate_model(model, test_loader, rul_max, device)
        test_losses.append(test_rmse)
        
        scheduler.step(test_rmse)
        
        # Save best model
        if test_rmse < best_test_loss:
            best_test_loss = test_rmse
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
        
        if verbose and (epoch + 1) % 20 == 0:
            tqdm.write(
                f"  Epoch {epoch+1}/{n_epochs} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Test RMSE: {test_rmse:.2f} cycles"
            )
    
    # Load best model
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, weights_only=True))
    
    return train_losses, test_losses


def evaluate_model(model, test_loader, rul_max, device='cpu'):
    """
    Evaluate model on test set, return RMSE in original RUL scale.
    """
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X = batch_X.to(device)
            pred = model(batch_X)
            
            # Denormalize
            pred_rul = pred.cpu().numpy().flatten() * rul_max
            true_rul = batch_Y.numpy().flatten() * rul_max
            
            all_preds.append(pred_rul)
            all_trues.append(true_rul)
    
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    
    rmse = np.sqrt(np.mean((all_preds - all_trues) ** 2))
    return rmse


def get_predictions(model, test_loader, rul_max, device='cpu'):
    """
    Get all predictions and true values from test loader.
    
    Returns:
        preds: np.array of predicted RUL (original scale)
        trues: np.array of true RUL (original scale)
    """
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X = batch_X.to(device)
            pred = model(batch_X)
            
            pred_rul = pred.cpu().numpy().flatten() * rul_max
            true_rul = batch_Y.numpy().flatten() * rul_max
            
            all_preds.append(pred_rul)
            all_trues.append(true_rul)
    
    return np.concatenate(all_preds), np.concatenate(all_trues)


def get_predictions_with_uncertainty(model, test_loader, rul_max, n_mc=50, device='cpu'):
    """
    Get MC Dropout predictions with uncertainty estimates.
    
    Returns:
        mean_preds: np.array [N], mean predicted RUL
        std_preds:  np.array [N], std of predicted RUL (uncertainty)
        trues:      np.array [N], true RUL
    """
    all_means = []
    all_stds = []
    all_trues = []
    
    for batch_X, batch_Y in test_loader:
        batch_X = batch_X.to(device)
        
        mean_pred, std_pred, _ = model.predict_with_dropout(
            batch_X, n_samples=n_mc
        )
        
        mean_rul = mean_pred.cpu().numpy().flatten() * rul_max
        std_rul = std_pred.cpu().numpy().flatten() * rul_max
        true_rul = batch_Y.numpy().flatten() * rul_max
        
        all_means.append(mean_rul)
        all_stds.append(std_rul)
        all_trues.append(true_rul)
    
    return (
        np.concatenate(all_means),
        np.concatenate(all_stds),
        np.concatenate(all_trues)
    )


def create_and_train_model(
    train_loader, test_loader, rul_max,
    loss_type='mse', linex_a=0.1,
    hidden_size=64, fc_dim=128, dropout=0.3,
    n_epochs=100, lr=1e-3,
    device='cpu', save_path=None, verbose=True
):
    """
    Convenience function: create model + choose loss + train.
    
    Args:
        loss_type: 'mse', 'linex', or 'phm08'
        linex_a:   LinEx shape parameter (only used if loss_type='linex')
    
    Returns:
        model, train_losses, test_losses
    """
    model = BiLSTM_RUL(
        input_size=1,
        hidden_size=hidden_size,
        fc_dim=fc_dim,
        dropout=dropout
    )
    
    if loss_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_type == 'linex':
        criterion = LinExLoss(a=linex_a)
    elif loss_type == 'phm08':
        from src.linex_loss import PHM08ScoreLoss
        criterion = PHM08ScoreLoss()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    train_losses, test_losses = train_model(
        model, train_loader, test_loader,
        criterion=criterion,
        rul_max=rul_max,
        n_epochs=n_epochs,
        lr=lr,
        device=device,
        save_path=save_path,
        verbose=verbose
    )
    
    return model, train_losses, test_losses
