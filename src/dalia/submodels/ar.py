# Copyright 2024-2025 DALIA authors. All rights reserved.
from tabulate import tabulate

import numpy as np

from dalia import sp, xp
from dalia.configs.submodels_config import ARSubModelConfig
from dalia.core.submodel import SubModel
from dalia.utils import add_str_header, get_host


class ARSubModel(SubModel):
    """Fit a stationary AR(p) model of arbitrary order p >= 1.

    The process x_t = sum_{j=1}^p phi_j x_{t-j} + eps_t is parametrized through
    its partial autocorrelations pacf1, ..., pacfp, each in (-1, 1), and its
    marginal precision tau, Var(x_t) = 1 / tau. Any admissible pacf vector maps
    to a stationary process; the AR coefficients follow from the
    Durbin-Levinson recursion and the innovation variance is

        sigma_eps^2 = prod_k (1 - pacf_k^2) / tau.

    The prior precision matrix is the exact stationary one, including the
    boundaries. With m_t = min(t - 1, p),

        (L x)_t = x_t - sum_{j=1}^{m_t} phi^(m_t)_j x_{t-j},
        D_tt    = prod_{k=1}^{m_t} (1 - pacf_k^2) / tau,

    the precision is Q = L^T D^{-1} L. The first p rows are the shorter
    conditional regressions of the stationary initial distribution, which makes
    the boundary entries exact. Q is banded with half-bandwidth min(p, n - 1).
    There is no restriction on the number of latent parameters; n = 1 gives
    Q = [tau].

    Only the parameter-independent structure (rows, columns) is cached. The
    full band is always stored, also where entries are numerically zero (e.g.
    zero pacfs), such that the sparsity pattern and the ordering of the data
    never change between evaluations. Entries are sorted row-major.
    """

    def __init__(
        self,
        config: ARSubModelConfig,
    ) -> None:
        """Initializes the model."""
        super().__init__(config)

        self.order: int = config.order
        self.bandwidth: int = min(self.order, self.n_latent_parameters - 1)

        # --- Sparsity pattern of the full band
        n = self.n_latent_parameters
        offsets = np.arange(-self.bandwidth, self.bandwidth + 1)
        rows = np.repeat(np.arange(n)[:, None], offsets.size, axis=1)
        cols = rows + offsets[None, :]

        self._mask = (cols >= 0) & (cols < n)
        self._row = xp.asarray(rows[self._mask], dtype=xp.int32)
        self._col = xp.asarray(cols[self._mask], dtype=xp.int32)

    @staticmethod
    def _pacf_to_ar_coefficients(pacf) -> list[np.ndarray]:
        """Durbin-Levinson recursion from partial autocorrelations to AR coefficients.

        Returns a list `levels`, where levels[m] holds the coefficients
        phi^(m)_1, ..., phi^(m)_m of the order-m conditional regression of x_t
        on x_{t-1}, ..., x_{t-m}; levels[0] is empty and levels[p] are the AR(p)
        coefficients.
        """
        levels = [np.empty(0, dtype=float)]
        for m, psi in enumerate(pacf, start=1):
            previous = levels[-1]
            current = np.empty(m, dtype=float)
            # every update reads the previous level only (no in-place recursion)
            current[: m - 1] = previous - psi * previous[::-1]
            current[m - 1] = psi
            levels.append(current)

        return levels

    def _check_hyperparameters(self, pacf: np.ndarray, tau: float) -> None:
        # a missing pacf is nan at this point
        if not np.all(np.isfinite(pacf)) or not np.all(np.abs(pacf) < 1.0):
            raise ValueError(
                f"Partial autocorrelations must be finite and in (-1, 1), got {pacf}."
            )
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError(
                f"Marginal precision tau must be finite and positive, got {tau}."
            )

    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix."""

        # kwargs expects hyperparameters in external scale
        pacf = np.array(
            [kwargs.get(f"pacf{k}") for k in range(1, self.order + 1)], dtype=float
        )
        tau = kwargs.get("tau")
        tau = np.nan if tau is None else float(tau)
        self._check_hyperparameters(pacf, tau)

        n, p, bw = self.n_latent_parameters, self.order, self.bandwidth

        levels = self._pacf_to_ar_coefficients(pacf)
        # conditional precisions 1 / D of the order-m regressions, m = 0, ..., p
        cond_prec = tau / np.concatenate(([1.0], np.cumprod(1.0 - pacf**2)))

        # bands[k][t] = Q[t, t + k]
        bands = [np.zeros(n - k) for k in range(bw + 1)]

        # boundary rows s < p: regression of order m_s = s
        for s in range(min(p, n)):
            c = np.concatenate(([1.0], -levels[s]))
            for k in range(s + 1):
                # row s contributes c[j - k] * c[j] to Q[s - j, s - j + k]
                bands[k][: s - k + 1] += cond_prec[s] * (c[: s - k + 1] * c[k:])[::-1]

        # interior rows s >= p: full order-p regression
        if n > p:
            c = np.concatenate(([1.0], -levels[p]))
            for k in range(p + 1):
                for j in range(k, p + 1):
                    bands[k][p - j : n - j] += cond_prec[p] * c[j - k] * c[j]

        values = np.zeros(self._mask.shape)
        for k in range(bw + 1):
            values[: n - k, bw + k] = bands[k]
            values[k:, bw - k] = bands[k]

        return sp.sparse.coo_matrix(
            (xp.asarray(values[self._mask]), (self._row, self._col)),
            shape=(n, n),
        )

    def derived_hyperparameters(
        self,
        theta_internal,
        cov_theta_internal,
        prior_hyperparameters,
        quantiles,
        n_samples: int = 10000,
        seed: int = 0,
        **kwargs,
    ) -> dict:
        """Compute the marginal distributions of the AR coefficients phi1, ..., phip.

        An AR coefficient is a function of several partial autocorrelations,
        its marginal distribution depends on their joint distribution and has
        no closed form. The Gaussian approximation of the partial
        autocorrelations in internal scale is sampled, the samples are rescaled
        to external scale and mapped to the AR coefficients.

        Parameters
        ----------
        theta_internal : NDArray
            Mode of the hyperparameters of the submodel in internal scale.
        cov_theta_internal : NDArray
            Covariance of the hyperparameters of the submodel in internal scale.
        prior_hyperparameters : list
            Prior hyperparameters of the submodel, they rescale the hyperparameters.
        quantiles : NDArray
            Quantiles to compute. If None, no quantiles are computed.
        n_samples : int
            Number of samples used for the computation of the marginal distributions.
        seed : int
            Seed of the samples used for the computation of the marginal distributions.
        kwargs : dict
            Options of the computation, they are submodel dependent.

        Returns
        -------
        dict
            Marginal distributions of the derived hyperparameters.
        """
        p = self.order

        rng = np.random.default_rng(seed)
        samples = rng.multivariate_normal(
            mean=get_host(theta_internal)[:p],
            cov=get_host(cov_theta_internal)[:p, :p],
            size=n_samples,
        )

        pacf = np.empty_like(samples)
        for k in range(p):
            pacf[:, k] = get_host(
                prior_hyperparameters[k].rescale_hyperparameters_to_internal(
                    xp.asarray(samples[:, k]), direction="backward"
                )
            )

        phi = np.array([self._pacf_to_ar_coefficients(sample)[-1] for sample in pacf])

        results = {}
        for j in range(p):
            pdf, edges = np.histogram(phi[:, j], bins=100, density=True)

            param_dict = {
                "mean_external": float(np.mean(phi[:, j])),
                "variance_external": float(np.var(phi[:, j])),
                "pdf_data": (0.5 * (edges[:-1] + edges[1:]), pdf),
            }

            if quantiles is not None:
                levels = get_host(quantiles).tolist()
                values = np.quantile(phi[:, j], levels).tolist()
                param_dict["quantiles"] = {
                    "levels": levels,
                    "external": {
                        "values": values,
                        "pairs": list(zip(levels, values)),
                    },
                }

            results[f"phi{j + 1}"] = param_dict

        return results

    def __str__(self) -> str:
        """String representation of the submodel."""
        str_representation = ""

        # --- Make the Submodel table ---
        values = [
            ["Submodel Type", self.submodel_type],
            ["Number of Latent Parameters", self.n_latent_parameters],
            ["Order", self.order],
        ]
        values += [
            [f"pacf{k}", f"{pacf:.3f}"]
            for k, pacf in enumerate(self.config.pacf, start=1)
        ]
        values += [["tau", f"{self.config.tau:.3f}"]]
        submodel_table = tabulate(
            values,
            tablefmt="fancy_grid",
            colalign=("left", "center"),
        )

        # Add the header title
        submodel_table = add_str_header(
            title=self.submodel_type.replace("_", " ").title(),
            table=submodel_table,
        )
        str_representation += submodel_table

        return str_representation
