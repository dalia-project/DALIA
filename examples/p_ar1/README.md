# The Model

### Overview

This example fits a Poisson likelihood model with an AR(1) latent process and one fixed effect (an intercept). The data consists of n counts generated as y ~ Poisson(E * exp(eta)), where eta = u + intercept, u is a zero mean AR(1) process with autocorrelation phi and marginal variance s2 (precision tau = 1/s2), and E is an exposure vector with entries sampled from {1, 2, 3}.

The AR(1) process is defined as

$$
u_1 \sim \mathcal{N}(0, s2), \qquad
u_t = \phi \, u_{t-1} + \varepsilon_t, \qquad
\varepsilon_t \sim \mathcal{N}\big(0 \; s2(1 - \phi^2)\big), \quad t = 2, \dots, n
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

The model contains two hyperparameters, theta = (phi, tau):

1. phi: autocorrelation of the AR(1) process, constrained to (0, 1), with a prior set through `ph_phi`.
2. tau: precision of the AR(1) process, constrained to be positive, with a prior set through `ph_tau`.

The Poisson likelihood has no hyperparameter of its own. The latent field has dimension n + 1: the n AR(1) states plus the intercept. The fixed effect uses a weak prior whose precision is `fixed_effects_prior_precision`.

### Scripts

`generate_data.py` builds the tridiagonal AR(1) precision matrix Q, samples the latent field u from the corresponding multivariate normal distribution, adds the intercept, draws the exposures E and the Poisson counts y, and writes all files needed by the model: the design matrices `inputs_ar1/a.npz` and `inputs_regression/a.npz`, the observations `y.npy`, the exposures `e.npy`, and the reference values `reference_outputs/theta_original.npy` and `reference_outputs/x_original.npy`.

`run.py` loads the generated inputs and the reference outputs, constructs the model from an `AR1SubModel` and a `RegressionSubModel` combined with a Poisson likelihood, and runs the hyperparameter optimization with a dense solver through `dalia.minimize()`. The initial values of phi and tau in `ar1_dict` differ from the values used to generate the data. It prints the estimated hyperparameters next to the true values and the error norms of the recovered latent field.

### Usage

Generate the data first, then run the inference:

```bash
python generate_data.py
python run.py
```

`generate_data.py` writes its output directories relative to the current working directory while `run.py` reads them relative to the script location, so execute both from inside the `p_ar1` directory.
