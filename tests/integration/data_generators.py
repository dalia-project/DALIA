"""Synthetic data generators for the integration tests.

Every generator writes the inputs of a model in the directory layout expected
by the DALIA submodels, together with the values used to generate the data in
`reference_outputs/`. The data is small on purpose, the tests have to be fast.
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from dalia.configs import submodels_config
from dalia.submodels import SpatialSubModel, SpatioTemporalSubModel
from dalia.utils import get_host


# --- Finite element matrices -------------------------------------------------
def planar_mesh(nx: int, ny: int, length: float = 1.0):
    """Regular triangulation of the square [0, length]^2."""
    xs = np.linspace(0.0, length, nx)
    ys = np.linspace(0.0, length, ny)
    nodes = np.array([[x, y] for y in ys for x in xs])

    triangles = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = j * nx + i
            n10, n01, n11 = n00 + 1, n00 + nx, n00 + nx + 1
            triangles.append([n00, n10, n11])
            triangles.append([n00, n11, n01])

    return nodes, np.array(triangles)


def spatial_fem_matrices(nodes, triangles):
    """Lumped mass matrix c0 and stiffness matrices g1, g2, g3 (P1 elements)."""
    ns = nodes.shape[0]
    c0 = np.zeros(ns)
    rows, cols, vals = [], [], []

    for tri in triangles:
        p = nodes[tri]
        edges = np.array([p[2] - p[1], p[0] - p[2], p[1] - p[0]])
        area = 0.5 * abs(edges[0, 0] * edges[1, 1] - edges[0, 1] * edges[1, 0])
        c0[tri] += area / 3.0
        local_g1 = edges @ edges.T / (4.0 * area)
        for a in range(3):
            for b in range(3):
                rows.append(tri[a])
                cols.append(tri[b])
                vals.append(local_g1[a, b])

    c0_inv = sp.diags(1.0 / c0)
    g1 = sp.csc_matrix((vals, (rows, cols)), shape=(ns, ns))
    g2 = g1 @ c0_inv @ g1
    g3 = g2 @ c0_inv @ g1
    # Remove the round-off asymmetry of the products
    g2 = 0.5 * (g2 + g2.T)
    g3 = 0.5 * (g3 + g3.T)

    # The model updates the values of Q prior in place, which requires
    # matrices in canonical format (sorted indices, no explicit zeros)
    matrices = [sp.diags(c0).tocsc(), g1, sp.csc_matrix(g2), sp.csc_matrix(g3)]
    for mat in matrices:
        mat.eliminate_zeros()
        mat.sort_indices()

    return matrices


def temporal_fem_matrices(nt: int, dt: float = 1.0):
    """Lumped mass m0, boundary matrix m1 and stiffness m2 of a 1D mesh."""
    m0 = np.full(nt, dt)
    m0[[0, -1]] = dt / 2.0

    m1 = np.zeros(nt)
    m1[[0, -1]] = 0.5

    main = np.full(nt, 2.0 / dt)
    main[[0, -1]] = 1.0 / dt
    off = np.full(nt - 1, -1.0 / dt)
    m2 = sp.diags([off, main, off], [-1, 0, 1])

    return sp.diags(m0).tocsc(), sp.diags(m1).tocsc(), m2.tocsc()


def projection_matrix(nodes, nx: int, ny: int, locations):
    """Barycentric projection matrix of the mesh built by `planar_mesh`."""
    length = nodes[:, 0].max()
    hx, hy = length / (nx - 1), length / (ny - 1)
    rows, cols, vals = [], [], []

    for k, (x, y) in enumerate(locations):
        i = min(int(x / hx), nx - 2)
        j = min(int(y / hy), ny - 2)
        u, v = x / hx - i, y / hy - j
        n00 = j * nx + i
        n10, n01, n11 = n00 + 1, n00 + nx, n00 + nx + 1
        if u >= v:  # triangle (n00, n10, n11)
            tri, weights = [n00, n10, n11], [1.0 - u, u - v, v]
        else:  # triangle (n00, n11, n01)
            tri, weights = [n00, n11, n01], [1.0 - v, u, v - u]
        rows += [k] * 3
        cols += tri
        vals += weights

    return sp.csr_matrix((vals, (rows, cols)), shape=(len(locations), len(nodes)))


# --- Sampling ----------------------------------------------------------------
def sample_gmrf(Q, rng):
    """Sample from N(0, Q^-1) through a dense Cholesky factorization."""
    Q = get_host(Q.toarray()) if hasattr(Q, "toarray") else get_host(Q)
    L = np.linalg.cholesky(Q)
    return np.linalg.solve(L.T, rng.standard_normal(Q.shape[0]))


def _write_regression(input_dir: Path, n_obs: int, n_fixed_effects: int, rng, scale=1.0):
    """Intercept plus standard normal covariates."""
    a = scale * rng.standard_normal((n_obs, n_fixed_effects))
    a[:, 0] = 1.0
    input_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(input_dir / "a.npz", sp.csr_matrix(a))
    return a


def _write_reference(data_dir: Path, theta, x):
    (data_dir / "reference_outputs").mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "reference_outputs" / "theta_original.npy", np.asarray(theta))
    np.save(data_dir / "reference_outputs" / "x_original.npy", np.asarray(x))


def _observe(eta, likelihood: str, prec_o: float, rng, data_dir: Path):
    if likelihood == "gaussian":
        y = eta + rng.normal(0.0, np.sqrt(1.0 / prec_o), size=eta.shape)
    elif likelihood == "poisson":
        e = rng.choice([1, 2, 3], size=eta.shape)
        np.save(data_dir / "e.npy", e)
        y = rng.poisson(e * np.exp(eta))
    else:
        raise ValueError(f"Likelihood not supported: {likelihood}")
    np.save(data_dir / "y.npy", y)


# --- Generators --------------------------------------------------------------
def generate_regression_data(
    data_dir: Path,
    likelihood: str,
    n_obs: int,
    beta,
    prec_o: float = None,
    seed: int = 0,
):
    """Regression model: eta = A beta."""
    rng = np.random.default_rng(seed)
    beta = np.asarray(beta, dtype=float)
    scale = 1.0 if likelihood == "gaussian" else 0.3

    a = _write_regression(data_dir / "inputs", n_obs, len(beta), rng, scale)
    _observe(a @ beta, likelihood, prec_o, rng, data_dir)

    theta = [] if prec_o is None else [prec_o]
    _write_reference(data_dir, theta, beta)


def write_spatial_inputs(input_dir: Path, nx: int, ny: int, n_obs: int, rng):
    nodes, triangles = planar_mesh(nx, ny)
    c0, g1, g2, g3 = spatial_fem_matrices(nodes, triangles)
    a = projection_matrix(nodes, nx, ny, rng.uniform(0.0, 1.0, size=(n_obs, 2)))

    input_dir.mkdir(parents=True, exist_ok=True)
    for name, mat in zip(("c0", "g1", "g2", "g3", "a"), (c0, g1, g2, g3, a)):
        sp.save_npz(input_dir / f"{name}.npz", mat)
    return a


def write_spatio_temporal_inputs(
    input_dir: Path, nx: int, ny: int, nt: int, n_obs_per_step: int, rng
):
    nodes, triangles = planar_mesh(nx, ny)
    c0, g1, g2, g3 = spatial_fem_matrices(nodes, triangles)
    m0, m1, m2 = temporal_fem_matrices(nt)
    # Every node is observed at every time step, together with random locations.
    # The structured solver requires the same sparsity pattern in all the blocks.
    n_random = n_obs_per_step - len(nodes)
    if n_random < 0:
        raise ValueError("n_obs_per_step must be at least the number of mesh nodes.")
    # The latent field is ordered time step by time step
    a = sp.block_diag(
        [
            projection_matrix(
                nodes,
                nx,
                ny,
                np.vstack((nodes, rng.uniform(0.0, 1.0, size=(n_random, 2)))),
            )
            for _ in range(nt)
        ]
    ).tocsr()
    a.eliminate_zeros()

    input_dir.mkdir(parents=True, exist_ok=True)
    matrices = (c0, g1, g2, g3, m0, m1, m2, a)
    for name, mat in zip(("c0", "g1", "g2", "g3", "m0", "m1", "m2", "a"), matrices):
        sp.save_npz(input_dir / f"{name}.npz", mat)
    return a


def generate_spatial_data(
    data_dir: Path,
    nx: int,
    ny: int,
    n_obs: int,
    r_s: float,
    sigma_e: float,
    beta,
    prec_o: float,
    seed: int = 0,
):
    """Gaussian spatial model, latent field ordered as [spatial, regression]."""
    rng = np.random.default_rng(seed)
    beta = np.asarray(beta, dtype=float)

    a_s = write_spatial_inputs(data_dir / "inputs_spatial", nx, ny, n_obs, rng)
    a_r = _write_regression(data_dir / "inputs_regression", n_obs, len(beta), rng)

    # The prior precision matrix comes from the submodel, at the true theta
    spatial = SpatialSubModel(
        config=submodels_config.parse_config(
            {
                "type": "spatial",
                "input_dir": f"{data_dir}/inputs_spatial",
                "spatial_domain_dimension": 2,
                "r_s": r_s,
                "sigma_e": sigma_e,
                "ph_s": {"type": "gaussian", "mean": r_s, "precision": 0.5},
                "ph_e": {"type": "gaussian", "mean": sigma_e, "precision": 0.5},
            }
        )
    )
    u = sample_gmrf(spatial.construct_Q_prior(r_s=r_s, sigma_e=sigma_e), rng)

    _observe(a_s @ u + a_r @ beta, "gaussian", prec_o, rng, data_dir)
    _write_reference(data_dir, [r_s, sigma_e, prec_o], np.concatenate((u, beta)))


def sample_spatio_temporal_field(
    input_dir: Path, r_s: float, r_t: float, sigma_st: float, rng
):
    """Sample the spatio-temporal field from the prior of the submodel."""
    spatio_temporal = SpatioTemporalSubModel(
        config=submodels_config.parse_config(
            {
                "type": "spatio_temporal",
                "input_dir": f"{input_dir}",
                "spatial_domain_dimension": 2,
                "r_s": r_s,
                "r_t": r_t,
                "sigma_st": sigma_st,
                "manifold": "plane",
                "ph_s": {"type": "gaussian", "mean": r_s, "precision": 0.5},
                "ph_t": {"type": "gaussian", "mean": r_t, "precision": 0.5},
                "ph_st": {"type": "gaussian", "mean": sigma_st, "precision": 0.5},
            }
        )
    )
    Q = spatio_temporal.construct_Q_prior(r_s=r_s, r_t=r_t, sigma_st=sigma_st)
    return sample_gmrf(Q, rng)


def generate_spatio_temporal_data(
    data_dir: Path,
    likelihood: str,
    nx: int,
    ny: int,
    nt: int,
    n_obs_per_step: int,
    r_s: float,
    r_t: float,
    sigma_st: float,
    beta,
    prec_o: float = None,
    seed: int = 0,
):
    """Spatio-temporal model, latent field ordered as [spatio_temporal, regression]."""
    rng = np.random.default_rng(seed)
    beta = np.asarray(beta, dtype=float)
    scale = 1.0 if likelihood == "gaussian" else 0.3

    st_dir = data_dir / "inputs_spatio_temporal"
    a_st = write_spatio_temporal_inputs(st_dir, nx, ny, nt, n_obs_per_step, rng)
    a_r = _write_regression(
        data_dir / "inputs_regression", a_st.shape[0], len(beta), rng, scale
    )
    u = sample_spatio_temporal_field(st_dir, r_s, r_t, sigma_st, rng)

    _observe(a_st @ u + a_r @ beta, likelihood, prec_o, rng, data_dir)

    theta = [r_s, r_t, sigma_st] + ([] if prec_o is None else [prec_o])
    _write_reference(data_dir, theta, np.concatenate((u, beta)))


def generate_coregional_spatio_temporal_data(
    data_dir: Path,
    nx: int,
    ny: int,
    nt: int,
    n_obs_per_step: int,
    r_s,
    r_t,
    prec_o,
    sigmas,
    lambda_0_1: float,
    betas,
    seed: int = 0,
):
    """Bivariate coregional model with Gaussian likelihoods.

    The fields of the two variates are v0 = sigma_0 u0 and
    v1 = lambda_0_1 v0 + sigma_1 u1, where u0 and u1 are unit variance
    spatio-temporal fields. Each model is ordered as [spatio_temporal, regression].
    """
    rng = np.random.default_rng(seed)

    a_st, a_r, u = [], [], []
    for k in range(2):
        model_dir = data_dir / f"model_{k + 1}"
        st_dir = model_dir / "inputs_spatio_temporal"
        a_st.append(write_spatio_temporal_inputs(st_dir, nx, ny, nt, n_obs_per_step, rng))
        a_r.append(
            _write_regression(
                model_dir / "inputs_regression", a_st[k].shape[0], len(betas[k]), rng
            )
        )
        u.append(sample_spatio_temporal_field(st_dir, r_s[k], r_t[k], 0.0, rng))

    v0 = np.exp(sigmas[0]) * u[0]
    v1 = lambda_0_1 * v0 + np.exp(sigmas[1]) * u[1]

    x = []
    for k, v in enumerate((v0, v1)):
        beta = np.asarray(betas[k], dtype=float)
        eta = a_st[k] @ v + a_r[k] @ beta
        _observe(eta, "gaussian", prec_o[k], rng, data_dir / f"model_{k + 1}")
        x.append(np.concatenate((v, beta)))

    theta = [
        r_s[0], r_t[0], prec_o[0],
        r_s[1], r_t[1], prec_o[1],
        sigmas[0], sigmas[1], lambda_0_1,
    ]  # fmt: skip
    _write_reference(data_dir, theta, np.concatenate(x))
