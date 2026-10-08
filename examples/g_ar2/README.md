# The Model

### Overview

This example fits a Gaussian likelihood model with an AR(2) latent process and one fixed effect (an intercept). The data consists of n = 1000 observations generated as y = eta + noise, where eta = u + intercept, u is a zero mean stationary AR(2) process with marginal variance s2 = 5 (precision tau = 1/s2), and the observation noise is Gaussian with precision 100.

The AR(2) process x_t = phi1 x_{t-1} + phi2 x_{t-2} + eps_t is parametrized through its partial autocorrelations (pacf1, pacf2), each in (-1, 1). The AR coefficients follow as phi2 = pacf2 and phi1 = pacf1 (1 - pacf2), which guarantees a stationary process and therefore a positive definite precision matrix for every admissible value of the hyperparameters. This is the same parametrization R-INLA uses for its `ar` latent model (see References). The data is generated with pacf1 = 0.6 and pacf2 = -0.4, i.e. phi1 = 0.84 and phi2 = -0.4.

The model contains four hyperparameters, theta = (pacf1, pacf2, tau, prec_o):

1. pacf1: first partial autocorrelation (equal to the lag-1 autocorrelation) of the AR(2) process, in (-1, 1), with a Beta(2, 2) prior scaled to the support (-1, 1).
2. pacf2: second partial autocorrelation (equal to phi2) of the AR(2) process, in (-1, 1), with a Beta(2, 2) prior scaled to the support (-1, 1).
3. tau: marginal precision of the AR(2) process, constrained to be positive, with a Gamma(2, 1) prior.
4. prec_o: precision of the Gaussian observation noise, with a weak Gamma(2, 0.01) prior.

The beta prior accepts a `support` field. The default (0, 1) is the standard beta distribution; `[-1, 1]` scales it to the full range of a partial autocorrelation, and alpha = beta gives a prior symmetric around 0.

The latent field has dimension 1001: the 1000 AR(2) states plus the intercept. The fixed effect uses a weak prior precision of 0.001.

### Scripts

`generate_data.py` builds the AR(2) precision matrix Q from the exact stationary distribution, samples the latent field u through a Cholesky factorization of Q, adds the intercept (value 2) and Gaussian observation noise, and writes all files needed by the model: the design matrices `inputs_ar2/a.npz` and `inputs_regression/a.npz`, the observations `y.npy`, and the reference values `reference_outputs/theta_original.npy` and `reference_outputs/x_original.npy`. It also solves the conditional system directly as a sanity check of the generated data.

`run.py` loads the generated inputs and the reference outputs, constructs the model from an `AR2SubModel` and a `RegressionSubModel` combined with a Gaussian likelihood, and runs the full DALIA inference with a dense solver. Initial values are set away from the truth (pacf1 = 0.3, pacf2 = 0.1, tau = 3). It prints the estimated hyperparameters, the covariance of theta, the mean of the fixed effect, error norms of the reconstructed linear predictor eta, a comparison of the marginal variances of the latent parameters against a dense reference, and quantiles of the marginal posteriors of pacf1 and pacf2. It also plots the prior of tau and the marginal posterior distributions of the hyperparameters.

### Usage

Generate the data first, then run the inference:

```bash
python generate_data.py
python run.py
```

Both scripts resolve paths relative to their own location, so they can be executed from any directory.

### References

The parametrization of the AR(2) process by its partial autocorrelations and marginal precision follows R-INLA and the time series literature:

- R-INLA, [Autoregressive model of order p (AR(p))](https://inla.r-inla-download.org/r-inla.org/doc/latent/ar.pdf), latent model documentation. States the motivation directly: the AR coefficients have "severe non-linear constraints" for stationarity, so the model is re-parametrized by the partial autocorrelations. Its footnote gives the p = 2 map used here, psi1 = phi1 / (1 - phi2) and psi2 = phi2.
- S. E. Heaps (2022), [Enforcing stationarity through the prior in vector autoregressions](https://arxiv.org/abs/2004.09455), *Journal of Computational and Graphical Statistics*. Section 1 gives a short, readable account of the univariate case: Barndorff-Nielsen and Schou (1973) establish a bijection between the stationary AR(p) coefficients and the first p partial autocorrelations, which "are interpretable and only constrained to lie in the Cartesian product space (-1, 1)^p", and a uniform prior on the stationary region induces a closed-form prior on the partial autocorrelations.

