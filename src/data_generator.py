"""
Simulation Data Generator for High-Dimensional Mediation Analysis
"""

import numpy as np
import pandas as pd


def generate_simulation_data(n_samples, p_mediators, censoring_rate, seed=42,
                                   alpha_scale=1.0, beta_scale=1.0):
    """
    Generate simulation data for high-dimensional mediation analysis with survival outcome.
    """
    np.random.seed(seed)

    z_dim = 6

    # Generate exposure X and covariates Z from standard normal distribution
    X = np.random.normal(0, 1, (n_samples, 1))
    Z = np.random.normal(0, 1, (n_samples, z_dim))


    # True coefficient values for the first 10 active mediators
    # Configuration 3: balanced positive and negative effects
    TRUE_ALPHA = np.array([0.6, -0.5, 0.4, -0.55, 0.45, 0.7, -0.45, 0.5, 0.75, 0.6])
    TRUE_BETA = np.array([0.6, -0.5, 0.4, -0.55, 0.45, 0.7, -0.45, 0.5, 0.75, 0.65])

    # Apply scaling factors if specified (for weak signal scenarios)
    TRUE_ALPHA = TRUE_ALPHA * alpha_scale
    TRUE_BETA = TRUE_BETA * beta_scale

    # Initialize coefficient vectors (only first 10 are non-zero)
    alpha = np.zeros(p_mediators)
    beta = np.zeros(p_mediators)
    alpha[:10] = TRUE_ALPHA
    beta[:10] = TRUE_BETA
    gamma = 0.5  # Direct effect of X on T

    # AR(1) correlation structure for mediator errors
    rho = 0.25
    sigma_e = np.zeros((p_mediators, p_mediators))
    for i in range(p_mediators):
        for j in range(p_mediators):
            sigma_e[i, j] = rho ** abs(i - j)


    # Generate correlated errors via Cholesky decomposition
    L = np.linalg.cholesky(sigma_e)
    Z_indep = np.random.normal(0, 1, (n_samples, p_mediators))
    e = Z_indep @ L.T

    def g_k_nonlinear(z):
        """
        Nonlinear confounding function for mediator generation.
        """
        z1, z2, z3, z4, z5, z6 = z[:, 0], z[:, 1], z[:, 2], z[:, 3], z[:, 4], z[:, 5]

        term1 = 0.1 * z1 * z2
        term2 = 0.2 * np.cos(np.pi * z3**2 * z4) * z5
        term3 = 1 / (z6 + 5)

        return term1 + term2 + term3

    # Generate mediators: M_k = alpha_k * X + g_k(Z) + e_k
    M = np.zeros((n_samples, p_mediators))
    chunk_size = min(1000, p_mediators)

    for i in range(0, p_mediators, chunk_size):
        end_idx = min(i + chunk_size, p_mediators)
        chunk_len = end_idx - i

        # Compute g_k(Z) for all mediators in chunk
        gZ_batch = np.zeros((n_samples, chunk_len))
        for idx, k in enumerate(range(i, end_idx)):
            gZ_batch[:, idx] = g_k_nonlinear(Z)

        # Only first 10 mediators have non-zero alpha
        chunk_end = min(10, end_idx)
        active_count = chunk_end - i
        alpha_chunk = alpha[i:chunk_end]

        X_flat = X.flatten()
        alpha_X_part = np.zeros((n_samples, chunk_len))

        for idx in range(active_count):
            alpha_X_part[:, idx] = X_flat * alpha_chunk[idx]

        M[:, i:end_idx] = alpha_X_part + gZ_batch + e[:, i:end_idx]

    def g_nonlinear_survival(z):
        """
        Nonlinear confounding function for survival outcome.
        """
        z1, z2, z3, z4, z5, z6 = z[:, 0], z[:, 1], z[:, 2], z[:, 3], z[:, 4], z[:, 5]

        log_arg = z3**2
        log_term = np.log(log_arg)

        cube_arg = z4 * z5
        cube_term = np.cbrt(cube_arg)

        inner = (z1**2 * z2**3 + log_term + cube_term + np.exp(z6 / 2))
        result = (1/8) * inner - 3
        return result

    gZ_T = g_nonlinear_survival(Z)

    # Cox model linear predictor: sum(beta_k * M_k) + gamma * X + g(Z)
    lin_pred = M @ beta + gamma * X[:, 0] + gZ_T


    # Generate survival times from exponential distribution
    U = np.random.uniform(0, 1, n_samples)
    T = -np.log(1 - U) * np.exp(-lin_pred)


    # Calibrate censoring threshold c_0 to achieve target censoring rate
    # c_0 values are empirically determined for CR=0.2 and CR=0.4
    c_0 = 50 if censoring_rate == 0.4 else 300  # CR=0.4 or CR=0.2

    # Generate uniform censoring times on [0, c_0]
    C = np.random.uniform(0, c_0, n_samples)
    obs_time = np.minimum(T, C)
    event = (T <= C).astype(int)

    actual_censor = 1 - event.mean()

    # Check for extreme values
    extreme_ratio = np.sum(T > 100) / n_samples

    # Assemble data matrices
    features = np.hstack([X, M, Z])
    y = np.column_stack([obs_time, event])

    # Store true parameters for evaluation
    true_params = {
        'alpha_true': alpha,
        'beta_true': beta,
        'gamma_true': gamma,
        'active_indices': list(range(10)),
        'censoring_rate': censoring_rate,
        'c_0': c_0,
        'actual_censoring_rate': actual_censor,
        'uniform_censoring': True,
        'explicit_g': True,
        'x_dim': 1,
        'z_dim': 6,
        'm_dim': p_mediators,
        'model_type': 'Cox',
        'has_correlation': True,
        'T_statistics': {
            'min': float(T.min()),
            'median': float(np.median(T)),
            'max': float(T.max()),
            'mean': float(T.mean()),
            'std': float(T.std())
        }
    }

    return (pd.DataFrame(features), pd.DataFrame(y), true_params)
