import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def plot_prior_hp(param_name, theta_interval, prior_hp, log=False):
    """
    Plot prior distribution of a hyperparameter.

    All priors are in log-scale. Therefore exponeniate unless log=True.

    Parameters
    ----------
    param_name : str
        Name of the hyperparameter.
    theta_interval: tuple of float
        Interval (min, max) for plotting the prior.
    prior_hp : PriorHyperparameters
        Prior hyperparameter object.
    log : bool, optional
        Whether to plot in log-scale or original scale. Default is False.


    Note
    ----
    If log is True, the plot will be in log-scale. Otherwise, it will be in the original scale.


    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The figure and axes objects containing the plot.
    """

    if theta_interval[0] == 0:
        theta_interval = (1e-6, theta_interval[1])

    theta_vals = np.linspace(theta_interval[0], theta_interval[1], 200)
    prior_vals = np.array([prior_hp.evaluate_log_prior(theta) for theta in theta_vals])

    if log:
        xlabel = f"{param_name}"
        ylabel = "Log Prior Density"
    else:
        prior_vals = np.exp(prior_vals)
        xlabel = f"{param_name}"
        ylabel = "Prior Density"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(theta_vals, prior_vals, "b-", linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Prior Distribution of {param_name}")
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_marginal_distributions_hp(marginals_hp):
    """Plot marginal distributions of hyperparameters in both internal and external parametrizations."""

    # Get all hyperparameters
    hyperparams = marginals_hp["hyperparameters"]
    n_params = len(hyperparams)

    # Create subplot grid: n_params rows, 2 columns (internal left, external right)
    fig, axes = plt.subplots(n_params, 2, figsize=(15, 5 * n_params))

    # Handle case of single parameter
    if n_params == 1:
        axes = axes.reshape(1, -1)

    # Quantile colors and labels
    colors = ["#DEB887", "#DEB887", "darkred", "#DEB887", "#DEB887"]
    labels = ["2.5%", "25%", "50%", "75%", "97.5%"]

    for row, (param_name, param_data) in enumerate(hyperparams.items()):
        # Get internal parameters
        mean_internal = param_data["mean_internal"]
        var_internal = param_data["variance_internal"]
        std_internal = np.sqrt(var_internal)

        # Get external parameters
        mean_external = param_data["mean_external"]
        var_external = param_data["variance_external"]
        theta_external, pdf_external = param_data["pdf_data"]

        # Get quantiles
        quantile_pairs_internal = param_data["quantiles"]["internal"]["pairs"]
        quantile_pairs_external = param_data["quantiles"]["external"]["pairs"]

        # ===== LEFT PLOT: INTERNAL PARAMETRIZATION =====
        ax_left = axes[row, 0]

        # Create internal distribution (Gaussian)
        x_internal = np.linspace(
            mean_internal - 4 * std_internal, mean_internal + 4 * std_internal, 100
        )
        pdf_internal = norm.pdf(x_internal, loc=mean_internal, scale=std_internal)

        # Plot internal PDF
        ax_left.plot(
            x_internal, pdf_internal, "b-", linewidth=2, label="PDF (Internal)"
        )

        # Mark internal mean
        ax_left.axvline(
            mean_internal,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean = {mean_internal:.3f}",
        )

        # Mark internal quantiles
        for i, (prob, q_val) in enumerate(quantile_pairs_internal):
            if i < len(labels):
                ax_left.axvline(
                    q_val,
                    color=colors[i],
                    linestyle=":",
                    linewidth=2,
                    label=f"{labels[i]} = {q_val:.3f}",
                )

        ax_left.set_xlabel(f"{param_name} (internal scale)")
        ax_left.set_ylabel("PDF")
        ax_left.set_title(f"{param_name}: Internal Distribution (Gaussian)")
        ax_left.legend()
        ax_left.grid(True, alpha=0.3)

        # ===== RIGHT PLOT: EXTERNAL PARAMETRIZATION =====
        ax_right = axes[row, 1]

        # Plot external PDF
        ax_right.plot(theta_external, pdf_external, "b-", linewidth=2, label="PDF")

        # Mark external mean
        ax_right.axvline(
            mean_external,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean = {mean_external:.3f}",
        )

        # Mark external quantiles
        for i, (prob, q_val) in enumerate(quantile_pairs_external):
            if i < len(labels):
                ax_right.axvline(
                    q_val,
                    color=colors[i],
                    linestyle=":",
                    linewidth=2,
                    label=f"{labels[i]} = {q_val:.3f}",
                )

        ax_right.set_xlabel(f"{param_name} ")
        ax_right.set_ylabel("PDF")
        ax_right.set_title(f"{param_name}: Marginal Distribution")
        ax_right.legend()
        ax_right.grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.show()

    return fig, axes


def plot_marginal_distributions_hp_external(marginals_hp, true_means=None):
    """Plot marginal distributions of hyperparameters in both internal and external parametrizations."""

    # Get all hyperparameters
    hyperparams = marginals_hp["hyperparameters"]
    n_params = len(hyperparams)

    # Create subplot grid: 1 row, n_params columns
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5))

    # Handle case of single parameter - make it iterable
    if n_params == 1:
        axes = [axes]

    # Quantile colors and labels
    colors = ["#DEB887", "#DEB887", "darkblue", "#DEB887", "#DEB887"]
    labels = ["2.5%", "25%", "50%", "75%", "97.5%"]

    for col, (param_name, param_data) in enumerate(hyperparams.items()):
        # Get external parameters
        mean_external = param_data["mean_external"]
        var_external = param_data["variance_external"]
        theta_external, pdf_external = param_data["pdf_data"]

        # Get quantiles
        quantile_pairs_external = param_data["quantiles"]["external"]["pairs"]

        # ===== PLOT: EXTERNAL PARAMETRIZATION =====
        ax_right = axes[col]

        # Plot external PDF
        ax_right.plot(theta_external, pdf_external, "b-", linewidth=2, label="PDF")

        # Mark external mean
        ax_right.axvline(
            mean_external,
            color="darkgreen",
            linestyle="--",
            linewidth=2,
            label=f"Mean = {mean_external:.3f}",
        )

        # Mark true mean if provided
        if true_means is not None and col < len(true_means):
            ax_right.axvline(
                true_means[col],
                color="red",
                linestyle="-",
                linewidth=2,
                label=f"True Mean = {true_means[col]:.3f}",
            )

        # Mark external quantiles
        for i, (prob, q_val) in enumerate(quantile_pairs_external):
            if i < len(labels):
                ax_right.axvline(
                    q_val,
                    color=colors[i],
                    linestyle=":",
                    linewidth=2,
                    label=f"{labels[i]} = {q_val:.3f}",
                )

        ax_right.set_xlabel(f"{param_name} ")
        ax_right.set_ylabel("PDF")
        ax_right.set_title(f"{param_name}: Marginal Distribution")
        ax_right.legend()
        ax_right.grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.show()

    return fig, axes
