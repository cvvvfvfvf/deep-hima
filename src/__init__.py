"""
High-Dimensional Mediation Analysis for Survival Data
"""

# Main pipeline
from .main import complete_mediation_analysis, print_analysis_summary

# Data generation
from .data_generator import generate_simulation_data

# Individual estimation steps
from .estimate_alpha import estimate_alpha
from .estimate_gamma import estimate_gamma, initialize
from .marginal_beta import marginal_beta_scan
from .mediator_screening import filter_score
from .variance_estimation import estimate_var_beta
from .significance_test import bonferroni_joint, print_bonferroni_summary

# Model components
from .models import (
    CoxMediationModel,
    fit_scad_cox,
    fit_nonlinear_confounding,
    coxph_loss,
    scad_penalty,
    standardize
)

__version__ = '1.0.0'

__all__ = [
    # Main
    'complete_mediation_analysis',
    'generate_simulation_data',

    # Estimation functions
    'estimate_alpha',
    'estimate_gamma',
    'marginal_beta_scan',
    'initialize',
    'filter_score',
    'estimate_var_beta',
    'bonferroni_joint',

    # Model classes and functions
    'CoxMediationModel',
    'fit_scad_cox',
    'fit_nonlinear_confounding',
    'coxph_loss',
    'scad_penalty',
    'standardize',
]
