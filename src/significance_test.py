"""
Bonferroni Joint Significance Testing for Mediation Effects
"""

import numpy as np
from scipy.stats import norm


def bonferroni_joint(alpha_A, std_alpha_A, beta_A, std_beta_A, A, alpha_level=0.05):
    """
    Perform Bonferroni-corrected joint significance test for mediation.
    """
    nA = len(alpha_A)

    # Alpha pathway tests
    t_alpha = alpha_A / std_alpha_A
    p_raw_alpha = 2 * (1 - norm.cdf(np.abs(t_alpha)))
    p_corr_alpha = np.minimum(p_raw_alpha * nA, 1.0)

    # Beta pathway tests
    t_beta = beta_A / std_beta_A
    p_raw_beta = 2 * (1 - norm.cdf(np.abs(t_beta)))
    p_corr_beta = np.minimum(p_raw_beta * nA, 1.0)

    # Joint significance: both pathways must be significant
    p_joint = np.maximum(p_corr_alpha, p_corr_beta)
    signif = p_joint < alpha_level

    # Collect statistics
    stats = {
        'n_mediators': nA,
        'alpha_level': alpha_level,
        'n_significant': np.sum(signif),
        'p_raw_alpha_range': [np.min(p_raw_alpha), np.max(p_raw_alpha)],
        'p_raw_beta_range': [np.min(p_raw_beta), np.max(p_raw_beta)],
        'p_joint_range': [np.min(p_joint), np.max(p_joint)],
        't_alpha_range': [np.min(t_alpha), np.max(t_alpha)],
        't_beta_range': [np.min(t_beta), np.max(t_beta)]
    }

    return p_joint, signif, stats