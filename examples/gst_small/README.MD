# gst_small: small Gaussian Spatio-Temporal Model

This example demonstrates a small-scale spatio-temporal regression model with a Gaussian likelihood. It is designed to illustrate the structure and inference workflow for models with both spatial and temporal dependencies, as well as fixed effects (regression coefficients).

## Model Overview
| Component                | Value         |
|--------------------------|--------------|
| Likelihood               | Gaussian     |
| Hyperparameters          | 4            |
| Latent parameters        | 466          |
| Observations             | 4600         |
| Spatial locations (ns)   | 92           |
| Time points (nt)         | 5            |
| Manifold                 | Plane        |
| Fixed effects            | 6            |
| Fixed effects prior prec.| 0.001        |

## Expected Output
Running this example will:
- Fit the spatio-temporal Gaussian regression model to the provided data
- Estimate posterior distributions for latent variables and hyperparameters
- Output summary statistics and diagnostics (e.g., posterior means, variances)
- Optionally, generate predictions and residuals for the observed data

## Usage
1. Ensure all dependencies are installed (see project root README for setup instructions).
2. Run the example script(s) in this directory to perform inference.
3. Review the output files and logs for results and diagnostics.

## Files
- `inputs/` — Input data for the model
- `outputs/` — Model outputs, including posterior summaries and diagnostics
- `README.MD` — This file

For more details on the model structure and inference algorithms, see the main project documentation.
