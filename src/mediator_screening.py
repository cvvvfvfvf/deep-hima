"""
Mediator Screening via Product-of-Coefficients
"""

import numpy as np


def filter_score(alpha_hat, beta_hat, n):
    """
    Screen mediators by product-of-coefficients score.
    """
    p = len(alpha_hat)

    # Compute screening scores
    score = np.abs(alpha_hat * beta_hat)

    # Determine screening threshold: d = floor(n / log(n))
    d = int(n / np.log(n))

    # Select top d mediators by score
    top_idx = np.argsort(score)[::-1][:d]
    B = top_idx.tolist()

    return B
