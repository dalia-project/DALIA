import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp

BASE_DIR: Path = Path(__file__).parent

if __name__ == "__main__":

    np.random.seed(7)
    n = 1000
    n_burn_in = 2000

    ## define priors
    s2 = 5
    tau = 1 / s2  # marginal precision
    # partial autocorrelations, each in (-1, 1) -> stationary AR(3)
    pacf = [0.6, -0.4, 0.3]
    # noise obs
    obs_noise_prec = 100
    theta_original = [*pacf, tau, obs_noise_prec]

    # partial autocorrelations -> AR coefficients (Durbin-Levinson recursion)
    phi = []
    for m, psi in enumerate(pacf, start=1):
        phi = [phi[j] - psi * phi[m - 2 - j] for j in range(m - 1)] + [psi]
    phi = np.array(phi)
    print("AR coefficients: ", phi)

    # marginal variance -> innovation variance
    s2_eps = s2 * np.prod(1 - np.array(pacf) ** 2)
    print("marginal variance: ", s2, ", innovation variance: ", s2_eps)

    # simulate x_t = phi1 x_{t-1} + phi2 x_{t-2} + phi3 x_{t-3} + eps_t and
    # discard a burn-in, such that u is a draw from the stationary process
    p = len(pacf)
    eps = np.random.normal(0, np.sqrt(s2_eps), size=n_burn_in + n)
    x = np.zeros(n_burn_in + n)
    for t in range(p, n_burn_in + n):
        x[t] = phi @ x[t - p : t][::-1] + eps[t]
    u = x[n_burn_in:]

    print("Sample u statistics - mean:", np.mean(u), "std:", np.std(u), ". Should be around sqrt(s2) =", np.sqrt(s2))

    intercept = 2

    x = np.concatenate((u, [intercept]))
    print("x: ", x[:10])

    os.makedirs(BASE_DIR / "reference_outputs", exist_ok=True)
    np.save(BASE_DIR / "reference_outputs" / "x_original.npy", x)
    np.save(BASE_DIR / "reference_outputs" / "theta_original.npy", theta_original)

    os.makedirs(BASE_DIR / "inputs_ar", exist_ok=True)
    np.save(BASE_DIR / "inputs_ar" / "x.npy", u)

    a_ar = sp.eye(n)
    sp.save_npz(BASE_DIR / "inputs_ar" / "a.npz", a_ar)

    a_regression = sp.csr_matrix(np.ones((n, 1)))
    os.makedirs(BASE_DIR / "inputs_regression", exist_ok=True)
    sp.save_npz(BASE_DIR / "inputs_regression" / "a.npz", a_regression)

    eta = a_ar @ u + intercept
    print("eta: ", eta[:6])

    noise = np.random.normal(0, np.sqrt(1 / obs_noise_prec), size=eta.shape)
    y = eta + noise
    np.save(BASE_DIR / "y.npy", y)

    print("y: ", y[:10])
