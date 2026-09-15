# High-Dimensional Mediation Analysis for Survival Outcomes

This repository contains the simulation code for the paper:

**"Identifying Damage Pathways Linking Sequence Composition to Storage Failure in DNA Data Storage via High-Dimensional Mediation Analysis"**

by Jingyi Li, Huaming Wu, and Haixiang Zhang

## Overview

This package implements a three-step high-dimensional mediation analysis framework for survival outcomes:

1. **Step 1: Mediator Screening** - Product-of-coefficients screening based on |α̂β̂| scores
2. **Step 2: SCAD-Penalized Estimation** - Sparse estimation via deep partially linear Cox model
3. **Step 3: Joint Significance Testing** - Bonferroni-corrected joint hypothesis testing

Key features:
- Deep neural networks for flexible covariate adjustment
- GPU acceleration (CUDA/Intel XPU supported)
- Cox proportional hazards model for time-to-event data
- High-dimensional setting (p >> n scenarios)

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/cvvvfvfvf/deep-hima.git
cd deep-hima

# Create virtual environment
conda create -n hdmed python=3.9
conda activate hdmed

# Install dependencies
pip install -r requirements.txt

# Verify installation
python examples/quick_start.py
```

## Quick Start

### Quick Test

After installation, verify everything works:

```bash
python examples/quick_start.py
```

This runs a quick test on a small simulated dataset and should complete in 1-2 minutes.

### Basic Usage

```python
from src.data_generator import generate_simulation_data
from src.main import complete_mediation_analysis

# Generate simulated data
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
    verbose=False
)

# View results
print(f"Selected mediators: {len(results['final_mediators_A'])}")
print(f"Significant mediators: {results['significant_count']}")
print(f"C-index (test): {results['c_index_test']:.4f}")
```

See `examples/quick_start.py` for a complete working example.

### Batch Simulation

Run multiple replications for performance evaluation:

```bash
python scripts/run_simulation.py --n-samples 500 --p-mediators 300 --n-replica 200 --censoring-rate 0.2
```

## Project Structure

```
deep-hima/
├── src/                       # Core source code
│   ├── main.py               # Main analysis pipeline
│   ├── data_generator.py     # Simulation data generation
│   ├── estimate_alpha.py     # α coefficient estimation (X→M)
│   ├── estimate_gamma.py     # γ and g(Z) estimation
│   ├── marginal_beta.py      # per-mediator β estimation (M→T)
│   ├── mediator_screening.py # Mediator screening
│   ├── variance_estimation.py # Variance estimation for β
│   ├── significance_test.py  # Bonferroni joint test
│   └── models/
│       ├── cox_mediation_model.py  # Cox mediation model
│       ├── cox_loss.py             # Cox partial likelihood
│       ├── linear_fit.py           # SCAD penalized estimation
│       └── nonlinear_fit.py        # Nonlinear confounding adjustment
├── scripts/                  # Executable scripts
│   └── run_simulation.py     # Batch simulation runner
├── examples/                 # Usage examples
│   └── quick_start.py        # 5-minute quick start
├── docs/                     # Documentation
│   └── QUICK_START.md        # Quick start guide
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── CITATION.cff              # Citation information
└── README.md                 # This file
```

## Method Overview

### Data Generation Model

**Mediator model:**
```
M_k = α_k X + g_k(Z) + ε_k,  k = 1,...,p
```

**Survival model:**
```
λ(t|X,M,Z) = λ_0(t) exp{γX + Σ β_k M_k + g(Z)}
```

Where:
- X: exposure (scalar)
- M: high-dimensional mediator vector (p-dimensional)
- Z: covariates (q-dimensional)
- T: survival time
- g(·), g_k(·): nonlinear functions approximated by DNNs

### Three-Step Procedure

**Step 1: Screening**
- Estimate α̂_k via deep partially linear model for mediator
- Estimate β̂_k via deep partially linear Cox model
- Compute screening score: S_k = |α̂_k β̂_k|
- Select top mediators based on S_k

**Step 2: SCAD Estimation**
- Joint estimation on selected mediators
- SCAD penalty for sparsity
- Deep neural network for g(Z)

**Step 3: Joint Testing**
- Test H_0: α_k β_k = 0 for each selected mediator
- Bonferroni correction for multiple testing
- Declare significance at level 0.05

## Simulation Settings

The paper's simulation studies use:

- Sample sizes: n ∈ {500, 1000}
- Mediator dimensions: p = 300
- Active mediators: 10 (known truth)
- Censoring rates: {20%, 40%}
- Covariate distributions: Normal, Exponential, Uniform, Mixed

Performance metrics:
- Statistical power (TP/10)
- False discovery rate (FP/(TP+FP))
- C-index for prediction
- Bias and MSE of mediation effects

## Reproducibility

All simulations use fixed random seeds for reproducibility. To reproduce the paper's results:

```bash
# Table 1 (Normal distribution, CR=20%)
python scripts/run_simulation.py --n-samples 500 --censoring-rate 0.2 --n-replica 200

# Table 1 (Normal distribution, CR=40%)
python scripts/run_simulation.py --n-samples 500 --censoring-rate 0.4 --n-replica 200
```

## GPU Acceleration

The package automatically detects and uses available GPUs:

```python
# Automatic device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Intel GPU (XPU) support
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = torch.device('xpu')
```

For large-scale simulations, GPU acceleration provides ~10-20× speedup.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{li2026identifying,
  title={Identifying Damage Pathways Linking Sequence Composition to Storage Failure in DNA Data Storage via High-Dimensional Mediation Analysis},
  author={Li, Jingyi and Wu, Huaming and Zhang, Haixiang},
  journal={Under Review},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: haixiang.zhang@tju.edu.cn

## Acknowledgments

This work was supported by the National Natural Science Foundation of China and Tianjin University.
