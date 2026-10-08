# generate synthetic regression dataset


import os

# add .. to the path
import sys

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix

sys.path.append("..")

np.random.seed(41)

path = os.path.dirname(__file__)

if __name__ == "__main__":
    n_observations = 20
    n_latent_parameters = 6

    z = np.random.normal(size=n_latent_parameters)

    prior_precision = 1e-3
    Sigma_prior = 1 / prior_precision * np.eye(n_latent_parameters)
    L_Sigma_prior = np.linalg.cholesky(Sigma_prior)
    x = L_Sigma_prior @ z
    a = sparse.random(n_observations, n_latent_parameters, density=0.5)

    theta_observations = 2.0
    print(f"theta_observations: {theta_observations}")
    theta_likelihood: dict = {"theta_observations": theta_observations}

    # generate x from a gaussian distribution of dimensions n_latent_parameters with mean 0 and precision exp(theta_observations)
    variance = 1 / theta_observations
    eta = a @ x
    y = np.random.normal(eta, scale=np.sqrt(variance), size=n_observations)
    print(f"x: {x}")
    # print(f"y: {y}")
    # print(f"a: {a}")

    a = csr_matrix(a)
    print("A: \n", a[:10, :n_latent_parameters].toarray())
    print("y: ", y[:10])

    # create a subfolder called inputs
    os.makedirs(f"{path}/inputs", exist_ok=True)

    # save the synthetic data
    np.save(f"{path}/y.npy", y)
    # save a as .npz
    sparse.save_npz(f"{path}/inputs/a.npz", a)

    # save original latent parameters
    np.save(f"{path}/reference_outputs/x_ref.npy", x)

    # save original hyperparameter theta
    np.save(f"{path}/reference_outputs/theta_ref.npy", theta_observations)
