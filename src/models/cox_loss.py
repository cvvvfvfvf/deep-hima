"""
Cox Partial Likelihood Loss and SCAD Penalty Functions

This module implements the negative partial log-likelihood for Cox proportional
hazards models and the SCAD (Smoothly Clipped Absolute Deviation) penalty for
regularization in high-dimensional settings.
"""

import torch
import numpy as np


def _make_riskset(time):
    """
    Construct risk set matrix for Cox partial likelihood computation.
    """
    assert time.dim() == 1, 'Expected 1D tensor for time'
    n = time.size(0)

    # Sort times in descending order (largest times first)
    _, indices = torch.sort(time, descending=True, stable=True)

    risk_set = torch.zeros(n, n, dtype=torch.bool)

    # For each individual in sorted order
    for i_org, i_sort in enumerate(indices):
        ti = time[i_sort].item()

        # Find all individuals with time >= ti (including ties)
        k = i_org
        while k < n and ti == time[indices[k]].item():
            k += 1

        # Mark all individuals at risk at time ti
        risk_set[i_sort, indices[:k]] = True

    return risk_set


def logsumexp_masked(risk_scores, mask, axis=1, keepdims=None):
    """
    Compute numerically stable log-sum-exp over masked entries.
    """
    mask_f = mask.type(risk_scores.dtype)

    risk_scores = torch.transpose(risk_scores, 0, 1)
    risk_scores_masked = torch.mul(risk_scores, mask_f)

    # Find maximum for numerical stability
    amax, _ = torch.max(risk_scores_masked, dim=axis, keepdim=True)

    # Shift, exponentiate, mask, and sum
    risk_scores_shift = risk_scores_masked - amax
    exp_masked = torch.mul(torch.exp(risk_scores_shift), mask_f)
    exp_sum = torch.sum(exp_masked, dim=axis, keepdim=True)

    # Add back the maximum
    output = amax + torch.log(exp_sum)
    return output


def coxph_loss_(event, riskset, predictions):
    """
    Compute Cox partial likelihood loss (internal function).
    """
    # Compute log(sum_{j in R(t_i)} exp(eta_j))
    rr = logsumexp_masked(predictions, riskset, axis=1)

    event = event.type(rr.dtype)

    # Negative log partial likelihood: delta * (log_risk_sum - prediction)
    losses = torch.mul(event, rr - predictions)
    loss = torch.mean(losses)
    return loss


def coxph_loss(predictions, outcome):
    """
    Compute Cox partial likelihood loss for training.
    """
    time = outcome[:, 1]
    event = torch.unsqueeze(outcome[:, 0], 1)
    risk_set = _make_riskset(time)
    return coxph_loss_(event, risk_set, predictions)


def scad_penalty(beta, a=3.7, lam=1.0):
    """
    SCAD (Smoothly Clipped Absolute Deviation) penalty value..
    """
    beta_abs = torch.abs(beta)

    # Create masks for the three regions
    mask1 = (beta_abs <= lam).float()
    mask2 = torch.logical_and(beta_abs > lam, beta_abs <= a * lam).float()
    mask3 = (beta_abs > a * lam).float()

    # Region 1: lambda * |beta|
    out1 = lam * beta_abs

    # Region 2: (2*a*lambda*|beta| - beta^2 - lambda^2) / (2*(a-1))
    out2 = (2.0 * a * lam * beta_abs - beta_abs ** 2 - lam ** 2) / (2.0 * (a - 1.0))

    # Region 3: constant (a+1)*lambda^2/2
    out3 = torch.ones_like(beta) * (a + 1.0) * lam * lam * 0.5

    # Combine regions
    out = mask1 * out1 + mask2 * out2 + mask3 * out3
    return torch.sum(out)


def coxph_loss_scad_like(predictions, outcome):
    """
    Vectorized Cox partial likelihood loss (alternative implementation).
    """
    predictions = torch.squeeze(predictions)
    n = predictions.size(0)
    delta = outcome[:, 1]

    # Clamp predictions for numerical stability
    pred_clamped = torch.clamp(predictions, -30.0, 30.0)
    haz = torch.exp(pred_clamped)

    # Compute risk set sums via reverse cumulative sum
    # risk_sum[i] = sum of hazards for all j with time >= time[i]
    rsk = torch.flip(torch.cumsum(torch.flip(haz, [0]), dim=0), [0])
    rsk = rsk.clamp(min=1e-15)  # Prevent log(0)

    # Negative log partial likelihood
    Loss = torch.sum(delta * predictions - delta * torch.log(rsk))
    return -Loss
