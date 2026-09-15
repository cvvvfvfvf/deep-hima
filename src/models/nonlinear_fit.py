"""
Nonlinear Component Fitting for Deep Partially Linear Cox Model
"""

import torch
from .cox_loss import coxph_loss_scad_like


def fit_nonlinear_confounding(x, offset, y, model, weight_decay, learning_rate, epoch):
    """
    Fit the nonlinear component g(Z) using a neural network.
    """
    optimizer = torch.optim.Adam(
        [{'params': model.parameters(), 'weight_decay': weight_decay}],
        lr=learning_rate
    )

    for i in range(epoch):
        # Forward pass: compute g(Z)
        pred = model(x)

        # Add linear offset: eta = g(Z) + beta'M + gamma*X
        pred = pred + offset

        # Compute negative Cox partial likelihood
        loss = coxph_loss_scad_like(pred, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
