"""
Alpha Coefficient Estimation for Exposure-to-Mediator Pathway
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import Adam
import time


def estimate_alpha(X, Z, M, n_epochs=500, patience=20, mediator_batch_size=50,
                   lr=0.001, weight_decay=0.001, hidden1=16, hidden2=8,
                   dropout=0.1, seed=None, cross_fit=False):
    """
    Estimate alpha coefficients for exposure-to-mediator pathway.
    """
    n, p = M.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    start_time = time.time()

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # 1. Split data on CPU first (80/20 train/validation for early stopping)
    indices = np.random.permutation(n)
    train_size = int(0.8 * n)
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    # Standardize covariates Z
    scaler_Z = StandardScaler()
    Z_scaled = scaler_Z.fit_transform(Z)

    # Split data
    Z_train_cpu, Z_val_cpu = Z_scaled[train_idx], Z_scaled[val_idx]

    # Convert to tensors
    Z_train = torch.tensor(Z_train_cpu, dtype=torch.float32).to(device)
    Z_val = torch.tensor(Z_val_cpu, dtype=torch.float32).to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, 1).to(device)
    Z_tensor = torch.tensor(Z_scaled, dtype=torch.float32).to(device)
    M_tensor = torch.tensor(M, dtype=torch.float32).to(device)

    # Cross-fitting fold split (only used when cross_fit=True)
    if cross_fit:
        perm_cf = np.random.permutation(n)
        folds = [perm_cf[: n // 2], perm_cf[n // 2:]]

        # Within each fold, reserve 10% for early stopping validation
        fold_splits = []
        for f in range(2):
            tr_ids = folds[1 - f]  # Train on the other fold
            n_v = max(int(0.1 * len(tr_ids)), 20)
            fold_splits.append({
                'test': torch.tensor(folds[f], dtype=torch.long, device=device),
                'train': torch.tensor(tr_ids[n_v:], dtype=torch.long, device=device),
                'val': torch.tensor(tr_ids[:n_v], dtype=torch.long, device=device),
            })

    # Initialize results
    alpha_hat = np.zeros(p)
    var_alpha = np.zeros(p)
    residuals = np.zeros((n, p))
    M_hat = np.zeros((n, p))
    sigma2_alpha = np.zeros(p)

    # Neural network for g_k(Z) estimation
    class EfficientDNN(nn.Module):
        """
        Feedforward neural network for confounding function g_k(Z).

        Architecture: Z -> Hidden1 (ReLU + Dropout) -> Hidden2 (ReLU + Dropout) -> Output
        """
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden2, 1)
            )

        def forward(self, z):
            return self.net(z)

    def train_gk(z_tr, m_tr, z_va, m_va):
        """
        Train a single g_k network with early stopping.
        """
        net = EfficientDNN(Z_tensor.shape[1]).to(device)
        optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(n_epochs):
            net.train()
            optimizer.zero_grad()
            pred_train = net(z_tr)
            loss_train = criterion(pred_train, m_tr)
            loss_train.backward()
            optimizer.step()

            # Validation
            net.eval()
            with torch.no_grad():
                loss_val = criterion(net(z_va), m_va).item()

            if loss_val < best_loss:
                best_loss = loss_val
                patience_counter = 0
                best_model_state = net.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        net.load_state_dict(best_model_state)
        net.eval()
        return net

    # Process mediators in batches
    total_batches = (p + mediator_batch_size - 1) // mediator_batch_size

    for batch_idx in range(total_batches):
        batch_start = batch_idx * mediator_batch_size
        batch_end = min((batch_idx + 1) * mediator_batch_size, p)


        for k in range(batch_start, batch_end):
            if not cross_fit:
                # ---- Standard estimation: 80/20 early stopping, full-sample (in-sample) prediction ----
                Mk_train = M_tensor[train_idx, k].view(-1, 1)
                Mk_val = M_tensor[val_idx, k].view(-1, 1)
                net = train_gk(Z_train, Mk_train, Z_val, Mk_val)
                with torch.no_grad():
                    gk_hat = net(Z_tensor).squeeze()
                del net
            else:
                # ---- Cross-fitting: each fold's ĝ_k from network trained on other fold (out-of-sample) ----
                gk_hat = torch.zeros(n, dtype=torch.float32, device=device)
                for fs in fold_splits:
                    net = train_gk(
                        Z_tensor[fs['train']], M_tensor[fs['train'], k].view(-1, 1),
                        Z_tensor[fs['val']], M_tensor[fs['val'], k].view(-1, 1)
                    )
                    with torch.no_grad():
                        gk_hat[fs['test']] = net(Z_tensor[fs['test']]).squeeze()
                    del net

            with torch.no_grad():
                # Adjusted mediator: M̃_k = M_k - ĝ_k(Z)
                M_hat_k = M_tensor[:, k] - gk_hat

                # Estimate alpha_k via OLS: alpha_k = (X'M̃_k) / (X'X)
                X_flat = X_tensor.squeeze()
                numerator = torch.sum(X_flat * M_hat_k)
                denominator = torch.sum(X_flat * X_flat)

                if denominator > 1e-8:
                    alpha_k = numerator / denominator
                else:
                    alpha_k = torch.tensor(0.0)

                alpha_hat[k] = alpha_k.cpu().item()

                # Compute residuals: ê_k = M̃_k - alpha_k * X
                residuals_k = M_hat_k - alpha_k * X_flat
                residuals[:, k] = residuals_k.cpu().numpy()
                M_hat[:, k] = M_hat_k.cpu().numpy()

                # Residual variance: σ²_k = (ê'ê) / (n - 1)
                residual_sum_squares = torch.sum(residuals_k ** 2)
                sigma2_k = residual_sum_squares.item() / (n - 1)  # Unbiased estimator
                sigma2_alpha[k] = sigma2_k

                # Variance of alpha_k: Var(alpha_k) = σ²_k / (X'X)
                if denominator > 1e-8 and sigma2_k > 1e-8:
                    var_alpha[k] = sigma2_k / denominator.item()
                else:
                    var_alpha[k] = np.inf

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    end_time = time.time()

    return alpha_hat, var_alpha, residuals, M_hat, sigma2_alpha
