"""
Complete High-Dimensional Mediation Analysis Pipeline
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from estimate_alpha import estimate_alpha
from estimate_gamma import estimate_gamma
from marginal_beta import marginal_beta_scan
from mediator_screening import filter_score
from models.cox_mediation_model import CoxMediationModel
from variance_estimation import estimate_var_beta
from significance_test import bonferroni_joint


def complete_mediation_analysis(data_x, data_y, true_params, verbose=False, device=None,
                                seed=None, alpha_kwargs=None, beta_kwargs=None,
                                refit_on_support=True, refit_gamma=False,
                                scad_params_override=None):
    """
    Run the three-step high-dimensional mediation analysis.
    """
    # Setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    alpha_kwargs = alpha_kwargs or {}
    beta_kwargs = beta_kwargs or {}

    # Extract dimensions
    x_dim = true_params['x_dim']
    m_dim = true_params['m_dim']
    z_dim = true_params['z_dim']
    active_indices = true_params.get('active_indices', list(range(10)))

    # Initialize default results
    default_results = {
        'selected_mediators_B': [],
        'final_mediators_A': [],
        'true_positive': 0,
        'false_positive': 0,
        'false_negative': 0,
        'true_negative': 0,
        'c_index_train': 0.5,
        'c_index_test': 0.5,
        'beta_final': np.zeros(m_dim),
        'significant_mediators': np.zeros(m_dim, dtype=bool),
        'alpha_estimates': np.zeros(m_dim),
        'compute_device': str(device)
    }

    results = default_results.copy()

    # Extract data components
    X = data_x.iloc[:, :x_dim].values
    M = data_x.iloc[:, x_dim:x_dim+m_dim].values
    Z = data_x.iloc[:, x_dim+m_dim:].values

    if isinstance(data_y, pd.DataFrame):
        T = data_y.iloc[:, 0].values
        Delta = data_y.iloc[:, 1].values
    else:
        T = data_y[:, 0]
        Delta = data_y[:, 1]

    # ==================== Stage 1: Initial Estimation ====================
    X_single = X[:, 0] if x_dim > 0 else np.zeros(X.shape[0])

    # Estimate alpha (X -> M)
    alpha_hat, var_alpha, residuals, M_hat, sigma2_alpha = estimate_alpha(
        X_single, Z, M, seed=seed, **alpha_kwargs
    )
    active_var_alpha = var_alpha[active_indices]

    # Partially linear Cox model for the exposure and covariate effects.
    # Supplies gamma_hat (offset fixed during SCAD) and g_hat(Z) (offset for
    # the mediator-by-mediator fits below).
    gamma_hat, gz_hat = estimate_gamma(
        X_single, Z, M, T, Delta,
        seed=(seed + 1 if seed is not None else None),
        return_gz=True, **beta_kwargs
    )

    # Estimate beta_k one mediator at a time, as specified for the screening
    # step. All p two-parameter problems are solved in one vectorized Newton
    # pass, with g_hat(Z) held fixed as a known offset.
    beta_hat, gamma_marginal, _marg_info = marginal_beta_scan(
        X_single, Z, M, T, Delta, offset=gz_hat,
        fit_gamma=True, verbose=verbose, device=device
    )

    # Screen mediators by |alpha * beta|
    B = filter_score(alpha_hat, beta_hat, data_x.shape[0])
    scores = np.abs(alpha_hat * beta_hat)
    active_in_B = [idx for idx in active_indices if idx in B]

    # ==================== Stage 2: SCAD Penalized Estimation ====================

    # Train/test split
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
        data_x, data_y, test_size=0.2, random_state=42
    )

    if isinstance(data_y_train, pd.DataFrame):
        T_train = data_y_train.iloc[:, 0].values
        Delta_train = data_y_train.iloc[:, 1].values
    else:
        T_train = data_y_train[:, 0]
        Delta_train = data_y_train[:, 1]

    # Degeneracy protection
    import os as _os_guard
    _MIN_SUPP = int(_os_guard.environ.get('MED_MIN_SUPPORT', '25'))
    _MIN_EV = int(_os_guard.environ.get('MED_MIN_EVENT_SUPPORT', '12'))

    if len(B) > 0 and _MIN_SUPP > 0:
        m_start_idx = x_dim
        m_end_idx = x_dim + m_dim
        _Mtr = data_x_train.iloc[:, m_start_idx:m_end_idx].values
        _dtr = np.asarray(Delta_train).astype(bool)
        _kept = []

        for _j in B:
            _col = _Mtr[:, _j]
            _nz = _col > (_col.min() + 1e-9)
            if int(_nz.sum()) >= _MIN_SUPP and int((_nz & _dtr).sum()) >= _MIN_EV:
                _kept.append(_j)

        B = _kept

    # Build filtered feature matrices
    m_start_idx = x_dim
    m_end_idx = x_dim + m_dim
    M_B_indices = [m_start_idx + i for i in B]
    M_B_train = data_x_train.iloc[:, M_B_indices].copy() if len(M_B_indices) > 0 else pd.DataFrame()
    M_B_test = data_x_test.iloc[:, M_B_indices].copy() if len(M_B_indices) > 0 else pd.DataFrame()

    x_train_filtered = pd.concat([
        data_x_train.iloc[:, :x_dim].reset_index(drop=True),
        M_B_train.reset_index(drop=True),
        data_x_train.iloc[:, m_end_idx:].reset_index(drop=True)
    ], axis=1)

    x_test_filtered = pd.concat([
        data_x_test.iloc[:, :x_dim].reset_index(drop=True),
        M_B_test.reset_index(drop=True),
        data_x_test.iloc[:, m_end_idx:].reset_index(drop=True)
    ], axis=1)

    # Prepare gamma
    if np.isscalar(gamma_hat):
        gamma_hat_array = np.full(x_dim, gamma_hat)
    else:
        gamma_hat_array = gamma_hat

    # SCAD parameters
    import os as _os
    _lam_env = _os.environ.get('SCAD_LAMBDA_GRID')
    _scad_lambda = [float(x) for x in _lam_env.split(',')] if _lam_env else [0.07]

    scad_params = {
        'scad_LAMBDA': _scad_lambda,
        'dnn_epoch': 50,
        'scad_dnn_epoch': 10,
        'neuron1': 8,
        'neuron2': 8,
        'weight_decay': 0.01,
        'learning_rate': 0.01,
        'dropout1': 0.5,
        'dropout2': 0.5
    }

    if scad_params_override:
        scad_params.update(scad_params_override)

    # Fit Cox mediation model
    model = CoxMediationModel(
        x_dim=x_dim,
        m_dim=len(B),
        z_dim=z_dim,
        gamma=gamma_hat_array,
        neuron1=scad_params['neuron1'],
        neuron2=scad_params['neuron2'],
        dropout1=scad_params['dropout1'],
        dropout2=scad_params['dropout2'],
        scad_LAMBDA=scad_params['scad_LAMBDA'],
        weight_decay=scad_params['weight_decay'],
        dnn_epoch=scad_params['dnn_epoch'],
        scad_dnn_epoch=scad_params['scad_dnn_epoch']
    )

    model.fit(x_train_filtered, data_y_train)

    # Select best model by EBIC
    bic_id, selected_lambda = model._select_best_model()
    beta_bic = model.BETA[:, bic_id]
    beta_bic_orig = model.BETA_ORIG[:, bic_id]

    # Full beta vector
    beta_full = np.zeros(m_dim)
    beta_full[B] = beta_bic_orig

    # Compute alpha*beta ranking
    ab_products = alpha_hat * beta_full
    sorted_indices = np.argsort(-np.abs(ab_products))

    # Performance metrics
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for i in range(m_dim):
        if i in active_indices:
            if beta_full[i] != 0:
                true_positive += 1
            else:
                false_negative += 1
        else:
            if beta_full[i] != 0:
                false_positive += 1
            else:
                true_negative += 1

    # C-index
    try:
        y_train_df = (data_y_train.reset_index(drop=True)
                      if isinstance(data_y_train, pd.DataFrame)
                      else pd.DataFrame(data_y_train).reset_index(drop=True))
        y_test_df = (data_y_test.reset_index(drop=True)
                     if isinstance(data_y_test, pd.DataFrame)
                     else pd.DataFrame(data_y_test).reset_index(drop=True))

        x_train_reset = x_train_filtered.reset_index(drop=True)
        x_test_reset = x_test_filtered.reset_index(drop=True)

        c_bic_train = model.score(x_train_reset, y_train_df)
        c_bic_test = model.score(x_test_reset, y_test_df)

    except Exception:
        c_bic_train = 0.5
        c_bic_test = 0.5

    # Support set
    A = [B[i] for i in range(len(B)) if beta_bic[i] != 0]

    # Optional refitting
    if refit_on_support and len(A) > 0:
        MA_col_idx = [m_start_idx + j for j in A]

        x_train_refit = pd.concat([
            data_x_train.iloc[:, :x_dim].reset_index(drop=True),
            data_x_train.iloc[:, MA_col_idx].reset_index(drop=True),
            data_x_train.iloc[:, m_end_idx:].reset_index(drop=True)
        ], axis=1)

        x_test_refit = pd.concat([
            data_x_test.iloc[:, :x_dim].reset_index(drop=True),
            data_x_test.iloc[:, MA_col_idx].reset_index(drop=True),
            data_x_test.iloc[:, m_end_idx:].reset_index(drop=True)
        ], axis=1)

        if refit_gamma:
            refit_x_dim = 0
            refit_m_dim = x_dim + len(A)
            refit_gamma_arr = np.zeros(0)
        else:
            refit_x_dim = x_dim
            refit_m_dim = len(A)
            refit_gamma_arr = gamma_hat_array

        model_refit = CoxMediationModel(
            x_dim=refit_x_dim,
            m_dim=refit_m_dim,
            z_dim=z_dim,
            gamma=refit_gamma_arr,
            neuron1=scad_params['neuron1'],
            neuron2=scad_params['neuron2'],
            dropout1=scad_params['dropout1'],
            dropout2=scad_params['dropout2'],
            scad_LAMBDA=[0.0],
            weight_decay=scad_params['weight_decay'],
            dnn_epoch=scad_params['dnn_epoch'],
            scad_dnn_epoch=scad_params['scad_dnn_epoch']
    )

        model_refit.fit(x_train_refit, data_y_train)
        refit_id, _ = model_refit._select_best_model()
        refit_coef = model_refit.BETA_ORIG[:, refit_id]

        if refit_gamma:
            gamma_hat_array = refit_coef[:x_dim]
            beta_full[A] = refit_coef[x_dim:]
        else:
            beta_full[A] = refit_coef

        try:
            y_train_df_r = (data_y_train.reset_index(drop=True)
                            if isinstance(data_y_train, pd.DataFrame)
                            else pd.DataFrame(data_y_train).reset_index(drop=True))
            y_test_df_r = (data_y_test.reset_index(drop=True)
                           if isinstance(data_y_test, pd.DataFrame)
                           else pd.DataFrame(data_y_test).reset_index(drop=True))

            c_bic_train = model_refit.score(x_train_refit.reset_index(drop=True), y_train_df_r)
            c_bic_test = model_refit.score(x_test_refit.reset_index(drop=True), y_test_df_r)

        except Exception:
            pass

    # ==================== Stage 3: Inference ====================

    std_beta_full = np.zeros(m_dim)
    var_beta_full = np.zeros(m_dim)
    std_beta_A = np.array([])
    var_beta = np.array([])

    X_train = data_x_train.iloc[:, :x_dim].values
    M_train = data_x_train.iloc[:, x_dim:x_dim+m_dim].values
    Z_train = data_x_train.iloc[:, x_dim+m_dim:].values

    X_single_train = X_train[:, 0] if x_dim > 0 else np.zeros(X_train.shape[0])

    if len(A) > 0:
        MA_train = M_train[:, A]
        beta_A = beta_full[A]

        std_beta_A, var_beta, sigma2_beta = estimate_var_beta(
            X_single_train, Z_train, MA_train, T_train, Delta_train, beta_A
        )

        std_beta_full[A] = std_beta_A
        var_beta_full[A] = var_beta

    # Active mediator variances
    active_var_beta = np.zeros(len(active_indices))
    active_std_beta = np.zeros(len(active_indices))

    for i, med_idx in enumerate(active_indices):
        if med_idx in A:
            pos_in_A = list(A).index(med_idx)
            active_var_beta[i] = var_beta[pos_in_A]
            active_std_beta[i] = std_beta_A[pos_in_A]
        else:
            active_var_beta[i] = np.nan
            active_std_beta[i] = np.nan

    # Joint significance testing
    p_joint_full = np.zeros(m_dim)
    significant_full = np.zeros(m_dim, dtype=bool)
    significant_count = 0

    if len(A) > 0:
        try:
            alpha_A = alpha_hat[A]
            std_alpha_A = np.sqrt(var_alpha[A])
            beta_A = beta_full[A]

            p_joint, signif, stats = bonferroni_joint(
                alpha_A, std_alpha_A, beta_A, std_beta_A, A
            )

            p_joint_full[A] = p_joint
            significant_full[A] = signif
            significant_count = np.sum(signif)

        except Exception:
            pass

    # Error metrics based on joint testing
    tp_joint = 0
    fp_joint = 0
    fn_joint = 0
    tn_joint = 0

    for i in range(m_dim):
        is_truly_active = (i in active_indices)
        is_significant = significant_full[i]

        if is_truly_active and is_significant:
            tp_joint += 1
        elif not is_truly_active and is_significant:
            fp_joint += 1
        elif is_truly_active and not is_significant:
            fn_joint += 1
        else:
            tn_joint += 1

    fdr_joint = fp_joint / max(tp_joint + fp_joint, 1)
    power_joint = tp_joint / len(active_indices) if len(active_indices) > 0 else 0

    # ==================== Final Results ====================

    results.update({
        'alpha_hat': alpha_hat,
        'var_alpha': active_var_alpha,
        'beta_hat': beta_hat,
        'beta_final': beta_full,
        'std_beta_final': std_beta_full,
        'gamma_hat': gamma_hat_array,
        'var_beta': active_var_beta,
        'screening_scores': scores,
        'selected_mediators_B': B,
        'final_mediators_A': A,
        'significant_mediators': significant_full,
        'significant_count': significant_count,
        'p_joint_full': p_joint_full,
        'active_in_B': active_in_B,
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'true_negative': true_negative,
        'tp_joint': tp_joint,
        'fp_joint': fp_joint,
        'fn_joint': fn_joint,
        'tn_joint': tn_joint,
        'fdr_joint': fdr_joint,
        'power_joint': power_joint,
        'fdr_scad': false_positive / max(false_positive + true_positive, 1),
        'selected_lambda': selected_lambda,
        'ab_products': ab_products,
        'ab_sorted_indices': sorted_indices,
        'c_index_train': c_bic_train,
        'c_index_test': c_bic_test,
        'model': model,
        'true_params': true_params,
        'alpha_estimates': alpha_hat
    })

    return results