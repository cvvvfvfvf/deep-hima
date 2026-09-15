"""
Exposure coefficient and nuisance function estimation.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError


def cox_loss(eta, t, delta):
    """
    Breslow negative partial log-likelihood, normalized by the event count.
    """
    eta = eta.view(-1)
    t = t.view(-1)
    delta = delta.view(-1)

    # Sort by time in descending order
    t_sorted, indices = torch.sort(t, descending=True)
    eta_sorted = eta[indices]
    delta_sorted = delta[indices]

    # Numerical stability: subtract max before exp
    eta_max = eta_sorted.max()
    exp_eta = torch.exp(eta_sorted - eta_max)

    # Cumulative sum for risk set
    cumulative_sum = torch.cumsum(exp_eta, dim=0)
    log_risk = torch.log(cumulative_sum + 1e-10) + eta_max

    # Partial likelihood
    pl = delta_sorted * (eta_sorted - log_risk)

    # Normalize by number of events
    n_events = torch.sum(delta_sorted)
    if n_events > 0:
        loss = -torch.sum(pl) / n_events
    else:
        loss = -torch.sum(pl)

    return loss


class coxnet(nn.Module):
    """
    Partially linear Cox model eta = gamma*X + g(Z), with g a DNN.
    """

    def __init__(self, z_dim, hidden_dims, dropout_rate):
        super().__init__()

        layers = []
        input_dim = z_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.g_net = nn.Sequential(*layers)
        self.gamma = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, X, Z):
        return self.gamma * X.view(-1) + self.g_net(Z).squeeze()


def estimate_gamma(X, Z, M, T, Delta,
                   n_epochs=500,
                   learning_rate=0.01,
                   validation_split=0.2,
                   weight_decay=0.2,
                   gamma_weight_decay=None,
                   hidden_dims=(100, 100),
                   dropout_rate=0.2,
                   use_best_val=False,
                   seed=None,
                   return_gz=False):
    """
    Fit eta = gamma X + g(Z) and return gamma_hat, optionally with g_hat(Z_i).
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    n_samples = len(X)
    z_dim = Z.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scaler_X = StandardScaler()
    scaler_Z = StandardScaler()
    scaler_M = StandardScaler()

    X_scaled = scaler_X.fit_transform(X.reshape(-1, 1)).flatten()
    Z_scaled = scaler_Z.fit_transform(Z)
    M_scaled = scaler_M.fit_transform(M)

    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * (1 - validation_split))
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    Z_tensor = torch.tensor(Z_scaled, dtype=torch.float32).to(device)
    M_tensor = torch.tensor(M_scaled, dtype=torch.float32).to(device)
    T_tensor = torch.tensor(T, dtype=torch.float32).to(device)
    Delta_tensor = torch.tensor(Delta, dtype=torch.float32).to(device)

    X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
    Z_train, Z_val = Z_tensor[train_idx], Z_tensor[val_idx]
    T_train, T_val = T_tensor[train_idx], T_tensor[val_idx]
    Delta_train, Delta_val = Delta_tensor[train_idx], Delta_tensor[val_idx]

    np.random.seed(42)
    gamma_init = initialize(
        X_train, Z_train, M_tensor[train_idx], T_train, Delta_train, M.shape[1]
    )

    model = coxnet(
        z_dim=z_dim,
        hidden_dims=list(hidden_dims),
        dropout_rate=dropout_rate
    ).to(device)

    with torch.no_grad():
        model.gamma.data = torch.tensor(gamma_init, dtype=torch.float32).to(device)

    if gamma_weight_decay is None:
        optimizer = Adam(model.parameters(), lr=learning_rate,
                         weight_decay=weight_decay)
    else:
        # gamma is structural; strong L2 on it introduces systematic bias.
        _g = [p for n, p in model.named_parameters() if n == 'gamma']
        _rest = [p for n, p in model.named_parameters() if n != 'gamma']
        optimizer = Adam([
            {'params': _rest, 'weight_decay': weight_decay},
            {'params': _g, 'weight_decay': gamma_weight_decay},
        ], lr=learning_rate)

    final_gamma = 0.0
    final_gz = None
    _best_val = float('inf')

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        loss_train = cox_loss(model(X_train.unsqueeze(1), Z_train),
                              T_train, Delta_train)
        loss_train.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            loss_val = cox_loss(model(X_val.unsqueeze(1), Z_val),
                                T_val, Delta_val).item()

        # g_hat must come from the same weights as the adopted gamma.
        adopt = loss_val < _best_val if use_best_val else epoch == n_epochs - 1
        if adopt:
            _best_val = min(_best_val, loss_val)
            final_gamma = model.gamma.item()
            if return_gz:
                with torch.no_grad():
                    final_gz = model.g_net(Z_tensor).view(-1).cpu().numpy()

    sd_X = float(scaler_X.scale_[0]) if scaler_X.scale_[0] > 1e-8 else 1.0
    final_gamma = final_gamma / sd_X

    if return_gz:
        return final_gamma, final_gz
    return final_gamma


def initialize(X, Z, M, T, Delta, p, max_mediators=50):
    """
    Starting value for gamma: average of Cox PH fits that each adjust for one
    mediator, so the result does not depend on which mediator is used.
    """
    to_np = lambda a: (a.detach().cpu().numpy()
                       if isinstance(a, torch.Tensor) else np.asarray(a))
    X, Z, M, T, Delta = (to_np(a) for a in (X, Z, M, T, Delta))

    if p <= max_mediators:
        mediator_indices = range(p)
    else:
        mediator_indices = np.sort(
            np.random.choice(p, size=max_mediators, replace=False)
        )

    gamma_vals = []

    for k in mediator_indices:
        mk = M[:, k]
        if (not np.all(np.isfinite(mk))) or np.nanstd(mk) < 1e-8:
            continue

        data = {'X': X, 'M_k': mk, 'time': T, 'status': Delta}
        for j in range(Z.shape[1]):
            data[f'Z{j}'] = Z[:, j]

        df = pd.DataFrame(data).replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < 20 or df['status'].sum() == 0 or df['M_k'].std() < 1e-8:
            continue

        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(df, duration_col='time', event_col='status',
                    show_progress=False)
        except (ConvergenceError, ValueError, np.linalg.LinAlgError):
            continue

        if 'X' in cph.summary.index:
            gamma_vals.append(cph.summary.loc['X', 'coef'])

    return float(np.mean(gamma_vals)) if gamma_vals else 0.0
