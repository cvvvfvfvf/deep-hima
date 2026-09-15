"""
Quick start example for high-dimensional mediation analysis.

This script demonstrates the basic usage of the package.
Run with: python examples/quick_start.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_generator import generate_simulation_data
from main import complete_mediation_analysis
import numpy as np

def main():
    # Generate simulation data
    data_x, data_y, true_params = generate_simulation_data(
        n_samples=500,
        p_mediators=300,
        censoring_rate=0.2,
        seed=42
    )

    # Run mediation analysis
    results = complete_mediation_analysis(
        data_x,
        data_y,
        true_params=true_params,
        verbose=False,  # Set to True for detailed output
        seed=42
    )

    # Display results

    print(f"\nScreening Stage:")
    print(f"  Selected mediators: {len(results['selected_mediators_B'])}/{true_params['m_dim']}")
    print(f"  True mediators in screening: {len(results['active_in_B'])}/{len(true_params['active_indices'])}")

    print(f"\nSCAD Stage:")
    print(f"  Non-zero mediators: {len(results['final_mediators_A'])}/{len(results['selected_mediators_B'])}")

    print(f"\nSignificance Testing:")
    print(f"  Significant mediators: {results['significant_count']}/{len(results['final_mediators_A'])}")

    print(f"\nPerformance Metrics:")
    print(f"  True Positive: {results['tp_joint']}/{len(true_params['active_indices'])}")
    print(f"  False Positive: {results['fp_joint']}")
    print(f"  FDR: {results['fdr_joint']:.4f}")
    print(f"  Power: {results['power_joint']:.4f}")
    print(f"  C-index (test): {results['c_index_test']:.4f}")

    # Show significant mediators
    if results['significant_count'] > 0:
        sig_idx = np.where(results['significant_mediators'])[0]
        print(f"\nSignificant Mediators (n={len(sig_idx)}):")
        print(f"{'Index':<8} {'α':<12} {'β':<12} {'α×β':<12} {'True?':<8}")
        print("-" * 60)
        for idx in sig_idx[:10]:  # Show first 10
            alpha = results['alpha_hat'][idx]
            beta = results['beta_final'][idx]
            is_true = '✓' if idx in true_params['active_indices'] else '✗'
            print(f"{idx:<8} {alpha:<12.4f} {beta:<12.4f} {alpha*beta:<12.4f} {is_true:<8}")
        if len(sig_idx) > 10:
            print(f"  ... and {len(sig_idx)-10} more")

if __name__ == "__main__":
    main()
