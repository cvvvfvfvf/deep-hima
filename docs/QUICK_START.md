# Quick Start Guide

This guide will help you get started with the high-dimensional mediation analysis package in 5 minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/cvvvfvfvf/deep-hima.git
cd deep-hima

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Generate Simulation Data

```python
from src.data_generator import generate_simulation_data

# Generate data with default parameters
data_x, data_y, true_params = generate_simulation_data(
    n_samples=500,      # Sample size
    p_mediators=300,    # Number of mediators
    censoring_rate=0.2, # Censoring rate (20% or 40%)
    seed=42             # Random seed for reproducibility
)
```

### 2. Run Mediation Analysis

```python
from src.main import complete_mediation_analysis

# Run complete analysis
results = complete_mediation_analysis(
    data_x,
    data_y,
    true_params=true_params,
    verbose=True,
    seed=42
)

# View results
print(f"Selected mediators (screening): {len(results['selected_mediators_B'])}")
print(f"Final mediators (SCAD): {len(results['final_mediators_A'])}")
print(f"Significant mediators: {results['significant_count']}")
print(f"C-index (test): {results['c_index_test']:.4f}")
```

### 3. Examine Significant Mediators

```python
import numpy as np

# Get significant mediators
sig_mediators = np.where(results['significant_mediators'])[0]
print(f"\nSignificant mediators: {sig_mediators}")

# Get their coefficients
for idx in sig_mediators:
    alpha = results['alpha_hat'][idx]
    beta = results['beta_final'][idx]
    print(f"Mediator {idx}: α={alpha:.4f}, β={beta:.4f}, α×β={alpha*beta:.4f}")
```

### 4. Evaluate Performance

```python
# Performance metrics
print(f"True Positive: {results['true_positive']}")
print(f"False Positive: {results['false_positive']}")
print(f"False Negative: {results['false_negative']}")
print(f"FDR: {results['fdr_joint']:.4f}")
print(f"Power: {results['power_joint']:.4f}")
```

## Complete Example

```python
# example_complete.py
from src.data_generator import generate_simulation_data
from src.main import complete_mediation_analysis
import numpy as np

# Step 1: Generate data
print("Generating simulation data...")
data_x, data_y, true_params = generate_simulation_data(
    n_samples=500,
    p_mediators=300,
    censoring_rate=0.2,
    seed=42
)

# Step 2: Run analysis
print("\nRunning mediation analysis...")
results = complete_mediation_analysis(
    data_x, data_y,
    true_params=true_params,
    verbose=True,
    seed=42
)

# Step 3: Report results
print(f"Screening: {len(results['selected_mediators_B'])}/{true_params['m_dim']}")
print(f"SCAD: {len(results['final_mediators_A'])}/{len(results['selected_mediators_B'])}")
print(f"Significant: {results['significant_count']}/{len(results['final_mediators_A'])}")
print(f"\nTrue mediators detected: {results['true_positive']}/{len(true_params['active_indices'])}")
print(f"FDR: {results['fdr_joint']:.4f}")
print(f"Power: {results['power_joint']:.4f}")
print(f"C-index: {results['c_index_test']:.4f}")
```

## Common Issues

### GPU not detected
If you have a GPU but it's not being used:
```python
import torch
print(torch.cuda.is_available())  # Should return True
```

### Out of memory
Reduce batch size or use CPU:
```python
results = complete_mediation_analysis(
    data_x, data_y,
    device=torch.device('cpu')
)
```
