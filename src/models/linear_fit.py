"""
Linear Fit Utilities for Deep Partially Linear Cox Model
"""

import torch


def standardize(X):
    """
    Standardize columns to zero mean and unit variance.
    """
    center = torch.mean(X, dim=0, keepdim=True)
    X_center = X - center
    scale = torch.pow(torch.mean(torch.pow(X_center, 2), dim=0, keepdim=True), 0.5)
    scale = torch.clamp(scale, min=1e-8)
    X_std = X_center / scale
    return {'X': X_std, 'center': center, 'scale': scale}


def wcrossprod(X, r, h, n, j):
    """
    Compute weighted cross-product for coordinate j.
    """
    return torch.dot(X[:, j], r * h)


def wsqsum(X, h, n, j):
    """
    Compute weighted sum of squares for coordinate j.
    """
    return torch.dot(X[:, j] ** 2, h)

def _scad_penalty_scalar(b, lam, gamma):
    """
    Scalar SCAD penalty value for a single coefficient.
    """
    b = abs(b)
    if b <= lam:
        return lam * b
    if b <= gamma * lam:
        return (2.0 * gamma * lam * b - b * b - lam * lam) / (2.0 * (gamma - 1.0))
    return (gamma + 1.0) * lam * lam / 2.0


def SCAD(z, lam, gamma, v):
    """
    SCAD proximal operator for coordinate descent update.
    """
    u = float(z)
    v = float(v)

    if v <= 0.0:
        return 0.0

    s = 1.0 if u >= 0.0 else -1.0
    au = abs(u)

    # Collect feasible stationary points (one per penalty segment)
    candidates = [0.0]

    # Region 1: |b| <= lam
    b = (au - lam) / v
    if 0.0 < b <= lam:
        candidates.append(b)

    # Region 2: lam < |b| <= gamma*lam
    den = v - 1.0 / (gamma - 1.0)
    if abs(den) > 1e-12:
        b = (au - gamma * lam / (gamma - 1.0)) / den
        if lam < b <= gamma * lam:
            candidates.append(b)

    # Region 3: |b| > gamma*lam
    b = au / v
    if b > gamma * lam:
        candidates.append(b)

    # Add segment endpoints (these are minimizers when middle segment is concave)
    candidates.append(lam)
    candidates.append(gamma * lam)

    # Evaluate objective at all candidates and return the best
    best = min(
        candidates,
        key=lambda b: -au * b + 0.5 * v * b * b + _scad_penalty_scalar(b, lam, gamma),
    )
    return s * best


def _compute_haz_h_r(eta, delta, offset):
    """
    Compute hazard, working weights, and working residuals from linear predictor.
    """
    eta_full = torch.clamp(eta, -30.0, 30.0)
    haz = torch.exp(eta_full)

    # Compute risk set sums via reverse cumulative sum
    rsk = torch.flip(torch.cumsum(torch.flip(haz, [0]), dim=0), [0])
    safe_rsk = rsk.clamp(min=1e-15)

    # Cumulative hazard contribution
    cum_delta_rsk = torch.cumsum(delta / safe_rsk, dim=0)
    h = cum_delta_rsk * haz
    safe_h = h.clamp(min=1e-15)

    # Working residuals
    r = torch.where(h > 0, (delta - h) / safe_h, torch.zeros_like(h))

    # Negative log partial likelihood
    loss = torch.sum(delta * eta_full - delta * torch.log(safe_rsk))

    return haz, h, safe_h, r, loss


def fit_scad_cox(X, delta, beta0, offset, lam, dfmax,
                     eps=1e-5, max_iter=10000, gamma=3.7):
    """
    Fit SCAD-penalized Cox model via coordinate descent.
    """
    n, p = X.shape
    offset = torch.squeeze(offset)

    # Initialize from warm start
    a = beta0.squeeze().clone().to(torch.double)  # shape (p,)
    eta = torch.matmul(X, a) + offset  # shape (n,)

    # Start with all variables active (efficient first sweep)
    e = torch.ones(p, dtype=torch.int8)

    beta = a.clone()  # Working coefficient vector

    tot_iter = 0
    while tot_iter < max_iter:

        # Recompute working weights and residuals at start of each sweep
        _, h, safe_h, r, loss_val = _compute_haz_h_r(eta, delta, offset)

        maxChange = 0.0

        # Coordinate descent sweep over active set
        for j in range(p):
            if not e[j]:
                continue

            xwr = wcrossprod(X, r, h, n, j)
            xwx = wsqsum(X, h, n, j)

            # Compute curvature term v = xwx / n
            v = float(xwx) / n

            # Skip near-constant columns (v < 1e-3) for numerical stability
            # These columns have negligible risk-weighted curvature and can
            # produce unstable updates (b = u/v explodes when v is tiny)
            if v < 1e-3:
                bj = float(beta[j])
                if bj != 0.0:
                    beta[j] = 0.0
                    eta = eta - bj * X[:, j]
                    r = r + bj * X[:, j]
                continue

            # Gradient term: u = xwr/n + v * beta[j]
            u = float(xwr) / n + v * float(beta[j])

            # SCAD proximal update
            new_b_val = SCAD(u, lam, gamma, v)

            shift = new_b_val - float(beta[j])
            if abs(shift) > 1e-15:
                beta[j] = new_b_val

                # Incrementally update linear predictor and working residuals
                eta = eta + shift * X[:, j]
                r = r - shift * X[:, j]

                # Track maximum standardized change
                chg = abs(shift) * (v ** 0.5)
                if chg > maxChange:
                    maxChange = chg

        tot_iter += 1

        # Check convergence
        if maxChange < eps:
            # Verify KKT conditions: check if any inactive variables violate optimality
            _, h, safe_h, r, _ = _compute_haz_h_r(eta, delta, offset)
            violations = 0
            for j in range(p):
                if not e[j]:
                    grad_j = abs(float(wcrossprod(X, r, h, n, j)) / n)
                    if grad_j > lam:
                        e[j] = 1
                        violations += 1

            if violations == 0:
                break  # True convergence: all KKT conditions satisfied

    # Return as (p, 1) tensor to match codebase convention
    return beta.unsqueeze(1)
