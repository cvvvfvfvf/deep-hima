"""
Batch Simulation Runner for High-Dimensional Mediation Analysis
"""

import sys
import os
import argparse
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import complete_mediation_analysis
from data_generator import generate_simulation_data_fixed


def run_simulation_study(n_samples=500, p_mediators=300, censoring_rate=0.4,
                        n_replica=100, active_indices=None):
    """
    Run multiple simulation replications and collect results.
    """
    # Device detection
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        device_name = "Intel GPU (XPU)"
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device('cpu')
        device_name = "CPU"

    if active_indices is None:
        active_indices = np.arange(10)

    # Storage for results
    all_results = []
    successful_runs = 0

    # Run simulations
    for run in range(1, n_replica + 1):
        
        if run % 10 == 0 or run == 1:
            print(f"Progress: {run}/{n_replica} replications completed", end='\r')

        try:
            # Generate data
            data_x, data_y, true_params = generate_simulation_data_fixed(
                n_samples=n_samples,
                p_mediators=p_mediators,
                censoring_rate=censoring_rate,
                seed=run
            )

            # Run analysis
            refit_support = os.environ.get('MED_REFIT_SUPPORT', '1') == '1'

            results = complete_mediation_analysis(
                data_x=data_x,
                data_y=data_y,
                true_params=true_params,
                device=device,
                seed=run,
                refit_on_support=refit_support
            )

            all_results.append(results)
            successful_runs += 1

        except Exception as e:
            print(f"Replication {run} failed: {e}")
        continue

    # Extract metrics
    selected_counts = [len(r['selected_mediators_B']) for r in all_results]
    final_counts = [len(r['final_mediators_A']) for r in all_results]
    significant_counts = [r['significant_count'] for r in all_results]

    # SCAD-based metrics
    tp_scad = [r['true_positive'] for r in all_results]
    fp_scad = [r['false_positive'] for r in all_results]
    fn_scad = [r['false_negative'] for r in all_results]
    fdr_scad = [r['false_positive'] / max(r['false_positive'] + r['true_positive'], 1)
                for r in all_results]

    # Joint test metrics
    tp_joint = [r['tp_joint'] for r in all_results]
    fp_joint = [r['fp_joint'] for r in all_results]
    fn_joint = [r['fn_joint'] for r in all_results]
    fdr_joint = [r['fdr_joint'] for r in all_results]
    power_joint = [r['power_joint'] for r in all_results]

    # C-index
    cindex_train = [r['c_index_train'] for r in all_results]
    cindex_test = [r['c_index_test'] for r in all_results]

    # Lambda selection
    selected_lambdas = [r['selected_lambda'] for r in all_results]

    # Coefficient estimates
    beta_estimates = []
    alpha_estimates = []
    beta_true_vals = []
    alpha_true_vals = []

    for r in all_results:
        if 'beta_final' in r and len(r['beta_final']) >= len(active_indices):
            beta_estimates.append(r['beta_final'][active_indices])
        if 'alpha_estimates' in r and len(r['alpha_estimates']) >= len(active_indices):
            alpha_estimates.append(r['alpha_estimates'][active_indices])
        if 'true_params' in r:
            if 'beta_true' in r['true_params']:
                beta_true_vals.append(r['true_params']['beta_true'][active_indices])
            if 'alpha_true' in r['true_params']:
                alpha_true_vals.append(r['true_params']['alpha_true'][active_indices])

    # Compute bias and RMSE
    if len(beta_estimates) > 0 and len(beta_true_vals) > 0:
        beta_est_arr = np.array(beta_estimates)
        beta_true_arr = np.array(beta_true_vals)
        beta_bias = np.mean(beta_est_arr - beta_true_arr, axis=0)
        beta_rmse = np.sqrt(np.mean((beta_est_arr - beta_true_arr)**2, axis=0))
    else:
        beta_bias = np.full(len(active_indices), np.nan)
        beta_rmse = np.full(len(active_indices), np.nan)

    if len(alpha_estimates) > 0 and len(alpha_true_vals) > 0:
        alpha_est_arr = np.array(alpha_estimates)
        alpha_true_arr = np.array(alpha_true_vals)
        alpha_bias = np.mean(alpha_est_arr - alpha_true_arr, axis=0)
        alpha_rmse = np.sqrt(np.mean((alpha_est_arr - alpha_true_arr)**2, axis=0))
    else:
        alpha_bias = np.full(len(active_indices), np.nan)
        alpha_rmse = np.full(len(active_indices), np.nan)

    # Per-mediator power (from joint test)
    significant_counter = np.zeros(p_mediators, dtype=int)
    for r in all_results:
        if 'significant_mediators' in r:
            sig_idx = np.where(r['significant_mediators'])[0]
            significant_counter[sig_idx] += 1

    mediator_power = significant_counter[active_indices] / successful_runs

    print(f"True Positive:    {np.mean(tp_joint):.2f} ± {np.std(tp_joint):.2f}")
    print(f"False Positive:   {np.mean(fp_joint):.2f} ± {np.std(fp_joint):.2f}")
    print(f"False Negative:   {np.mean(fn_joint):.2f} ± {np.std(fn_joint):.2f}")
    print(f"FDR (Joint):      {np.mean(fdr_joint):.3f} ± {np.std(fdr_joint):.3f}")
    print(f"Power (Joint):    {np.mean(power_joint):.3f} ± {np.std(power_joint):.3f}")
    print(f"C-index (train):  {np.mean(cindex_train):.4f} ± {np.std(cindex_train):.4f}")
    print(f"C-index (test):   {np.mean(cindex_test):.4f} ± {np.std(cindex_test):.4f}")

    for i, idx in enumerate(active_indices):
        print(f"M{idx+1:2d}:  {mediator_power[i]:.3f}")

    # Return summary dict
    results_summary = {
        'n_replica': n_replica,
        'successful_runs': successful_runs,
        'selected_counts': selected_counts,
        'final_counts': final_counts,
        'significant_counts': significant_counts,
        'tp_scad': tp_scad,
        'fp_scad': fp_scad,
        'fn_scad': fn_scad,
        'fdr_scad': fdr_scad,
        'tp_joint': tp_joint,
        'fp_joint': fp_joint,
        'fn_joint': fn_joint,
        'fdr_joint': fdr_joint,
        'power_joint': power_joint,
        'cindex_train': cindex_train,
        'cindex_test': cindex_test,
        'selected_lambdas': selected_lambdas,
        'mediator_power': mediator_power,
        'beta_bias': beta_bias,
        'beta_rmse': beta_rmse,
        'alpha_bias': alpha_bias,
        'alpha_rmse': alpha_rmse,
        'all_results': all_results
    }

    return results_summary


def main():
    """Command-line interface for simulation runner."""
    parser = argparse.ArgumentParser(
        description='Run high-dimensional mediation analysis simulations'
    )

    parser.add_argument('--n-samples', type=int, default=500,
                       help='Sample size per replication (default: 500)')
    parser.add_argument('--p-mediators', type=int, default=300,
                       help='Number of mediators (default: 300)')
    parser.add_argument('--censoring-rate', type=float, default=0.4,
                       help='Censoring rate (default: 0.4)')
    parser.add_argument('--n-replica', type=int, default=100,
                       help='Number of replications (default: 100)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed output per replication')

    args = parser.parse_args()

    # Run simulation study
    results = run_simulation_study(
        n_samples=args.n_samples,
        p_mediators=args.p_mediators,
        censoring_rate=args.censoring_rate,
        n_replica=args.n_replica,
        verbose=args.verbose
    )

    return results


if __name__ == '__main__':
    main()
