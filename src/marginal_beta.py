"""
Mediator-by-Mediator Marginal Beta Estimation for Step 1 Screening

"""

import numpy as np
import torch


def _tie_group_start(t_sorted):
    """Index of the first element of each tied block (t_sorted ascending)."""
    n = t_sorted.shape[0]
    idx = torch.arange(n, device=t_sorted.device)
    new_group = torch.ones(n, dtype=torch.bool, device=t_sorted.device)
    if n > 1:
        new_group[1:] = t_sorted[1:] != t_sorted[:-1]
    starts = torch.where(new_group, idx, torch.zeros_like(idx))
    return torch.cummax(starts, dim=0).values


def _suffix(A, group_start):
    """S(i, k) = sum over {j : Y_j >= Y_i} of A[j, k] (A sorted by time)."""
    rev = torch.flip(torch.cumsum(torch.flip(A, [0]), dim=0), [0])
    return rev[group_start]


def marginal_beta_scan(X, Z, M, T, Delta, offset,
                       max_iter=50, tol=1e-8, max_step=2.0,
                       fit_gamma=True, verbose=False, device=None,
                       min_support=25, min_event_support=12,
                       beta_cap=10.0):
    """
    Fit the mediator-specific Cox models and return marginal beta estimates.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X = np.asarray(X, dtype=np.float64).reshape(-1)
    M = np.asarray(M, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    Delta = np.asarray(Delta, dtype=np.float64).reshape(-1)
    offset = np.asarray(offset, dtype=np.float64).reshape(-1)

    n, p = M.shape
    n_events = int(Delta.sum())
    if verbose:
        print(f"Marginal beta scan: n={n}, p={p}, events={n_events}, "
              f"{'2' if fit_gamma else '1'} parameter(s) per mediator")

    if n_events == 0:
        return (np.zeros(p), np.zeros(p),
                {'converged': np.zeros(p, dtype=bool), 'n_iter': 0,
                 'reason': 'no events'})

    # Standardize for numerical stability; rescale back at the end.
    sd_X = X.std(ddof=0)
    sd_X = sd_X if sd_X > 1e-12 else 1.0
    Xs = (X - X.mean()) / sd_X

    mu_M = M.mean(axis=0)
    sd_M = M.std(axis=0, ddof=0)
    degenerate = sd_M < 1e-12
    sd_M_safe = np.where(degenerate, 1.0, sd_M)
    Ms = (M - mu_M) / sd_M_safe

    order = np.argsort(T, kind='mergesort')
    Xs, Ms = Xs[order], Ms[order]
    Ts, Ds, Os = T[order], Delta[order], offset[order]

    dt = torch.float64
    Xt = torch.tensor(Xs, dtype=dt, device=device)
    Mt = torch.tensor(Ms, dtype=dt, device=device)
    Tt = torch.tensor(Ts, dtype=dt, device=device)
    Dt = torch.tensor(Ds, dtype=dt, device=device)
    Ot = torch.tensor(Os, dtype=dt, device=device)

    gstart = _tie_group_start(Tt)
    n_tie_groups = int(torch.unique(Tt).numel())

    Xcol = Xt.view(-1, 1)
    Dcol = Dt.view(-1, 1)

    gam = torch.zeros(p, dtype=dt, device=device)
    bet = torch.zeros(p, dtype=dt, device=device)
    active = torch.ones(p, dtype=torch.bool, device=device)

    n_iter_done = 0
    for it in range(max_iter):
        n_iter_done = it + 1
        eta = bet.view(1, -1) * Mt + Ot.view(-1, 1)
        if fit_gamma:
            eta = eta + gam.view(1, -1) * Xcol
        eta = eta - eta.max(dim=0, keepdim=True).values
        w = torch.exp(eta)

        S0 = torch.clamp(_suffix(w, gstart), min=1e-300)
        Em = _suffix(w * Mt, gstart) / S0
        gm = (Dcol * (Mt - Em)).sum(dim=0)
        Hmm = (Dcol * (_suffix(w * Mt * Mt, gstart) / S0 - Em * Em)).sum(dim=0)

        if fit_gamma:
            Ex = _suffix(w * Xcol, gstart) / S0
            gx = (Dcol * (Xcol - Ex)).sum(dim=0)
            Hxx = (Dcol * (_suffix(w * Xcol * Xcol, gstart) / S0 - Ex * Ex)).sum(dim=0)
            Hxm = (Dcol * (_suffix(w * Xcol * Mt, gstart) / S0 - Ex * Em)).sum(dim=0)
            det = Hxx * Hmm - Hxm * Hxm
            bad = det.abs() < 1e-12
            det = torch.where(bad, torch.ones_like(det), det)
            step_g = (Hmm * gx - Hxm * gm) / det
            step_b = (Hxx * gm - Hxm * gx) / det
            step_g = torch.where(bad, torch.zeros_like(step_g), step_g)
            step_b = torch.where(bad, torch.zeros_like(step_b), step_b)
        else:
            bad = Hmm.abs() < 1e-12
            denom = torch.where(bad, torch.ones_like(Hmm), Hmm)
            step_b = torch.where(bad, torch.zeros_like(gm), gm / denom)
            step_g = torch.zeros_like(step_b)

        # Clip steps so near-separated columns cannot diverge in one update.
        step_g = torch.clamp(step_g, -max_step, max_step)
        step_b = torch.clamp(step_b, -max_step, max_step)
        step_g = torch.where(active, step_g, torch.zeros_like(step_g))
        step_b = torch.where(active, step_b, torch.zeros_like(step_b))

        gam = gam + step_g
        bet = bet + step_b

        active = torch.maximum(step_g.abs(), step_b.abs()) > tol
        if not bool(active.any()):
            break

    converged = (~active).cpu().numpy()
    beta = bet.cpu().numpy() / sd_M_safe
    beta[degenerate] = 0.0
    gamma = gam.cpu().numpy() / sd_X

    # Quasi-separation guard. After standardization, zero-count samples share
    # the column minimum, so effective support = entries above that minimum.
    low_support = np.zeros(p, dtype=bool)
    if min_support > 0 or min_event_support > 0:
        d_bool = Delta.astype(bool)
        nz = M > (M.min(axis=0) + 1e-9)
        supp = nz.sum(axis=0)
        ev_supp = (nz & d_bool[:, None]).sum(axis=0)
        low_support = (supp < min_support) | (ev_supp < min_event_support)
        beta[low_support] = 0.0

    exploded = np.isfinite(beta) & (np.abs(beta) > beta_cap) & (~low_support)
    beta[exploded] = 0.0
    beta[~np.isfinite(beta)] = 0.0

    if verbose:
        print(f"  Newton iterations: {n_iter_done}, "
              f"converged {int(converged.sum())}/{p} columns")
        if int(low_support.sum()) > 0:
            print(f"  quasi-separation guard: {int(low_support.sum())} columns "
                  f"with insufficient support set to zero")
        if int(exploded.sum()) > 0:
            print(f"  fallback clipping: {int(exploded.sum())} diverged "
                  f"columns set to zero")

    info = {
        'converged': converged,
        'n_iter': n_iter_done,
        'degenerate': degenerate,
        'low_support': low_support,
        'exploded': exploded,
        'n_tie_groups': n_tie_groups,
    }
    return beta, gamma, info
