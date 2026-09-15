"""
Variance Estimation for Beta Coefficients via Information Matrix
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler


def estimate_var_beta(X, Z, MA, T, Delta, beta_A):
    """
    Estimate standard errors for beta coefficients via information matrix.
    """
    n, pA = MA.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Standardize covariates
    scaler_Z = StandardScaler()
    Z_scaled = scaler_Z.fit_transform(Z)

    # Convert to tensors
    Z_tensor = torch.tensor(Z_scaled, dtype=torch.float32).to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    T_tensor = torch.tensor(T, dtype=torch.float32).to(device)
    Delta_tensor = torch.tensor(Delta, dtype=torch.float32).to(device)
    MA_tensor = torch.tensor(MA, dtype=torch.float32).to(device)

    # Extract event samples only
    event_mask = Delta == 1
    n_events = event_mask.sum()

    if n_events == 0:
        raise ValueError("No event samples available for information matrix estimation")

    event_mask_tensor = torch.tensor(event_mask, dtype=torch.bool).to(device)
    X_evt = X_tensor[event_mask_tensor].view(-1, 1)
    MA_evt = MA_tensor[event_mask_tensor]
    W_evt = torch.cat([X_evt, MA_evt], dim=1)  # (n_events, pA+1)

    # Input features: (T, Z)
    tz_input = torch.cat([T_tensor.view(-1, 1), Z_tensor], dim=1)
    tz_input_evt = tz_input[event_mask_tensor]  # (n_events, 1+dim(Z))

    # Auxiliary neural network: predicts W from (T, Z)
    class OptimizedAuxNet(nn.Module):
        """
        Neural network to estimate E[W | T, Z] where W = [X, M_A].

        Architecture: (T, Z) -> 64 (BN+ReLU+Dropout) -> 32 (BN+ReLU+Dropout) -> W
        """
        def __init__(self, tz_dim, out_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(tz_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, out_dim)
            )

        def forward(self, tz):
            return self.net(tz)

    aux_net = OptimizedAuxNet(1 + Z_tensor.shape[1], W_evt.shape[1]).to(device)

    optimizer = Adam(aux_net.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Train with validation-based early stopping
    # Split event samples into train/validation (80/20)
    n_evt_samples = int(W_evt.shape[0])
    perm = torch.randperm(n_evt_samples)
    n_val = max(int(0.2 * n_evt_samples), 10)
    val_sel, tr_sel = perm[:n_val], perm[n_val:]
    tz_tr, tz_val = tz_input_evt[tr_sel], tz_input_evt[val_sel]
    W_tr, W_val = W_evt[tr_sel], W_evt[val_sel]

    best_loss = float('inf')
    patience_counter = 0
    patience = 20
    best_model_state = None

    for epoch in range(500):
        aux_net.train()
        optimizer.zero_grad()
        pred = aux_net(tz_tr)
        loss = criterion(pred, W_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(aux_net.parameters(), max_norm=1.0)
        optimizer.step()

        # Validate
        aux_net.eval()
        with torch.no_grad():
            val_loss = criterion(aux_net(tz_val), W_val).item()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = aux_net.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience and epoch > 100:
            break

    # Load best model (by validation loss)
    aux_net.load_state_dict(best_model_state)

    # Estimate information matrix
    aux_net.eval()
    with torch.no_grad():
        pred_evt = aux_net(tz_input_evt)
        residual = W_evt - pred_evt  # (n_events, pA+1)

        # Information matrix I = (1/n) * sum(residual * residual')
        # Note: scaled by full sample size n, not n_events
        I_hat = torch.matmul(residual.T, residual) / n  # (pA+1, pA+1)
        I_hat_cpu = I_hat.cpu().numpy()

        # Invert to get covariance matrix
        try:
            Sigma_full = np.linalg.inv(I_hat_cpu)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            Sigma_full = np.linalg.pinv(I_hat_cpu)

        # Extract M block (remove first row and column for X)
        Sigma_M = Sigma_full[1:, 1:]  # (pA, pA)

        # Variance and standard error
        var_beta = np.diag(Sigma_M) / n  # Var(beta) = diag(I^{-1}) / n
        std_beta_A = np.sqrt(var_beta)  # SE(beta) = sqrt(Var(beta))

        # Residual variance (for diagnostics)
        residual_cpu = residual.cpu().numpy()
        sigma2_beta = np.var(residual_cpu, axis=0, ddof=2)[1:]  # Remove X part

    return std_beta_A, var_beta, sigma2_beta
