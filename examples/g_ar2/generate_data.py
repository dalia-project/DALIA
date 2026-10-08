import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, spsolve_triangular
from scipy.sparse import csc_matrix
from scipy.linalg import cholesky

BASE_DIR: Path = Path(__file__).parent

if __name__ == "__main__":

    np.random.seed(5)
    n = 1000

    ## define priors
    s2 = 5
    tau = 1 / s2
    # partial autocorrelations, each in (-1, 1) -> stationary AR(2)
    pacf1 = 0.6
    pacf2 = -0.4
    # noise obs
    obs_noise_prec = 100
    theta_original = [
        pacf1,
        pacf2,
        tau,
        obs_noise_prec,
    ]

    # partial autocorrelations -> AR coefficients
    phi2 = pacf2
    phi1 = pacf1 * (1 - pacf2)
    print("AR coefficients: phi1 =", phi1, ", phi2 =", phi2)

    # marginal variance -> innovation variance
    denom = s2 * (1 + phi2) * ((1 - phi2) ** 2 - phi1**2) / (1 - phi2)

    diag = [(1 + phi1**2 + phi2**2) / denom] * n
    diag[0] = diag[-1] = 1 / denom
    diag[1] = diag[-2] = (1 + phi1**2) / denom
    off_diag_1 = [-phi1 * (1 - phi2) / denom] * (n - 1)
    off_diag_1[0] = off_diag_1[-1] = -phi1 / denom
    off_diag_2 = [-phi2 / denom] * (n - 2)

    Q = sp.diags(
        [off_diag_2, off_diag_1, diag, off_diag_1, off_diag_2], [-2, -1, 0, 1, 2]
    )

    # Compute sparse Cholesky factorization: Q = L_upper.T @ L_upper
    # For pentadiagonal matrix, we can use dense Cholesky on small blocks or scipy
    Q_csc = Q.tocsc()

    print("Q shape:", Q.shape, "Q nnz:", Q.nnz)
    print("Q sparsity:", 100 * Q.nnz / (Q.shape[0] * Q.shape[1]), "%")
    print(Q.toarray()[:6, :6])

    # Method 1: Use dense Cholesky (for moderate sizes this is still efficient)
    Q_dense = Q.toarray()
    L_upper_dense = cholesky(Q_dense, lower=False)
    L_upper = csc_matrix(L_upper_dense)

    print("L_upper nnz:", L_upper.nnz, "L_upper sparsity:", 100 * L_upper.nnz / (L_upper.shape[0] * L_upper.shape[1]), "%")

    # Efficient sampling: generate z ~ N(0,I), then solve L.T @ u = z
    z = np.random.normal(0, 1, size=n)

    # cov(u) = L_upper^{-1} L_upper^{-T} = (L_upper.T @ L_upper)^{-1} = Q^{-1}
    u = spsolve_triangular(L_upper, z, lower=False)

    # Verify the sampling worked correctly
    print("Sample u statistics - mean:", np.mean(u), "std:", np.std(u), ". Should be around sqrt(s2) =", np.sqrt(s2))

    intercept = 2

    x = np.concatenate((u, [intercept]))
    print("x: ", x[:10])

    os.makedirs(BASE_DIR / "reference_outputs", exist_ok=True)
    np.save(BASE_DIR / "reference_outputs" / "x_original.npy", x)
    np.save(BASE_DIR / "reference_outputs" / "theta_original.npy", theta_original)

    os.makedirs(BASE_DIR / "inputs_ar2", exist_ok=True)
    np.save(BASE_DIR / "inputs_ar2" / "x.npy", u)

    a_ar2 = sp.eye(n)
    sp.save_npz(BASE_DIR / "inputs_ar2" / "a.npz", a_ar2)

    a_regression = sp.csr_matrix(np.ones((n, 1)))
    os.makedirs(BASE_DIR / "inputs_regression", exist_ok=True)
    sp.save_npz(BASE_DIR / "inputs_regression" / "a.npz", a_regression)

    eta = a_ar2 @ u + intercept

    print("eta: ", eta[:6])
    np.save(BASE_DIR / "inputs_ar2" / "x_original.npy", eta)

    noise = np.random.normal(0, np.sqrt(1 / obs_noise_prec), size=eta.shape)
    print("noise: ", noise[:10])
    y = eta + noise
    np.save(BASE_DIR / "y.npy", y)

    print("y: ", y[:10])

    Qprior = sp.block_diag([Q, sp.csr_matrix([[0.001]])])

    a = sp.hstack([a_ar2, a_regression])
    Qcond = Qprior + obs_noise_prec * a.T @ a
    print("Qcond: \n", Qcond.toarray()[:6, :6])

    b = obs_noise_prec * a.T @ y
    print("b: ", b[:10])
    x_est = spsolve(csc_matrix(Qcond), b)
    print("norm(x - x_est): ", np.linalg.norm(x - x_est))

    print("norm(eta - eta_est): ", np.linalg.norm(a @ x - a @ x_est))
    print("normalized norm(eta - eta_est): ", np.linalg.norm(a @ x - a @ x_est) / np.linalg.norm(a @ x))
