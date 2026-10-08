import os
import sys

import numpy as np
import scipy.sparse as sp

from scipy.stats import multivariate_normal, poisson

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    n = 1000

    ## define priors
    s2 = 1  # 0.7
    tau = 1 / s2
    ### note: phi between -1 and 1 for discrete timesteps
    # (doesn't make sense for negative in cts case)

    ## rescale beta prior 2 theta - 1
    phi = 0.9  # 0.9
    theta_original = [phi, tau]

    denom = s2 * (1 - phi**2)

    diag = [(1 + phi**2) / denom] * n
    diag[0] = diag[-1] = 1 / denom
    off_diag = [-phi / denom] * (n - 1)

    Q = sp.diags([diag, off_diag, off_diag], [0, -1, 1])
    L = np.linalg.cholesky(Q.toarray())
    Cov = np.linalg.inv(Q.toarray())

    print(Q.toarray()[:6, :6])
    print(np.linalg.inv(Q.toarray())[:6, :6])
    print(np.round(Q.toarray() @ np.linalg.inv(Q.toarray()), 6)[:6, :6])
    # exit()

    mv = multivariate_normal(mean=np.zeros(n), cov=Cov, seed=3)

    intercept = 2
    u = mv.rvs()
    print("u: ", u[:10])
    eta = u + intercept
    x = np.concatenate((u, [intercept]))

    os.makedirs("reference_outputs", exist_ok=True)
    np.save("reference_outputs/x_original.npy", x)
    np.save("reference_outputs/theta_original.npy", theta_original)

    x_initial = u + np.random.normal(0, 0.3, size=len(u))
    os.makedirs("inputs_ar1", exist_ok=True)
    np.save("inputs_ar1/x.npy", u)

    a_ar1 = sp.eye(n)
    sp.save_npz("inputs_ar1/a.npz", a_ar1)

    a_regression = sp.csr_matrix(np.ones((n, 1)))
    os.makedirs("inputs_regression", exist_ok=True)
    sp.save_npz("inputs_regression/a.npz", a_regression)

    print("eta: ", eta[:10])
    np.save("inputs_ar1/x_original.npy", eta)

    # sample with repitition
    E = np.random.choice([1, 2, 3], size=n, replace=True)
    # E = [1] * n
    np.save("e.npy", E)

    y = poisson.rvs(E * np.exp(eta), random_state=3)
    np.save("y.npy", y)

    print("y[:10]: ", y[:10])

    Qprior = sp.block_diag([Q, sp.csr_matrix([[0.001]])])
    # print("Qprior : \n", Qprior.toarray())

    # a = sp.hstack([a_ar1, a_regression])
    # Qcond = Qprior + obs_noise_prec * a.T @ a
    # print("Qcond: \n", Qcond.toarray())

    # b = obs_noise_prec * a.T @ y
    # # -xp.exp(theta) * (eta - y)
    # # beta_initial + np.linalg.solve(
    # #     Qconditional.toarray(), information_vector
    # # )
    # x_est = np.linalg.solve(Qcond.toarray(), b)
    # print("x_est: ", x_est)

    # print("eta est : ", a @ x_est)
    # print("eta :     ", a @ x)
