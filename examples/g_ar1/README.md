# The Model

### Overview

This example fits a Gaussian likelihood model with an AR(1) latent process and one fixed effect (an intercept). The data consists of n observations generated as y = eta + noise, where eta = u + intercept, u is a zero mean AR(1) process with autocorrelation phi and marginal variance s2 (precision tau = 1/s2), and the observation noise is Gaussian with precision obs_noise_prec.

The AR(1) process is defined as

$$
u_1 \sim \mathcal{N}(0, s2), \qquad
u_t = \phi \, u_{t-1} + \varepsilon_t, \qquad
\varepsilon_t \sim \mathcal{N}\big(0, \; s2(1 - \phi^2)\big), \quad t = 2, \dots, n
$$

which gives the tridiagonal precision matrix

$$
Q = \frac{1}{\texttt{denom}}
\begin{pmatrix}
1 & -\phi & & & \\
-\phi & 1 + \phi^2 & -\phi & & \\
& -\phi & 1 + \phi^2 & \ddots & \\
& & \ddots & \ddots & -\phi \\
& & & -\phi & 1
\end{pmatrix},
\qquad
\texttt{denom} = s2(1 - \phi^2).
$$

The model contains three hyperparameters, theta = (phi, tau, prec_o):

1. phi: autocorrelation of the AR(1) process, constrained to (0, 1), with a prior set through `ph_phi`.
2. tau: precision of the AR(1) process, constrained to be positive, with a prior set through `ph_tau`.
3. prec_o: precision of the Gaussian observation noise (obs_noise_prec in `generate_data.py`), with a prior set through `prior_hyperparameters` in the likelihood configuration.

The latent field has dimension n + 1: the n AR(1) states plus the intercept. The fixed effect uses a weak prior whose precision is `fixed_effects_prior_precision`.

### Scripts

`generate_data.py` builds the tridiagonal AR(1) precision matrix Q, samples the latent field u through a Cholesky factorization of Q, adds the intercept and Gaussian observation noise, and writes all files needed by the model: the design matrices `inputs_ar1/a.npz` and `inputs_regression/a.npz`, the observations `y.npy`, and the reference values `reference_outputs/theta_original.npy` and `reference_outputs/x_original.npy`. It also solves the conditional system directly as a sanity check of the generated data.

`run.py` loads the generated inputs and the reference outputs, constructs the model from an `AR1SubModel` and a `RegressionSubModel` combined with a Gaussian likelihood, and runs the full DALIA inference with a dense solver. It prints the estimated hyperparameters next to the true values, the covariance of theta, the mean of the fixed effect, error norms of the reconstructed linear predictor eta, a comparison of the marginal variances of the latent parameters against a dense reference, and quantiles of the marginal posterior of phi. It also plots the prior of tau and the marginal posterior distributions of the hyperparameters.

### Usage

Generate the data first, then run the inference:

```bash
python generate_data.py
python run.py
```

Both scripts resolve paths relative to their own location, so they can be executed from any directory.
