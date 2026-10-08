from pathlib import Path
from typing import Literal

import numpy as np
from scipy import sparse as sp

np.random.seed(5)

if __name__ == "__main__":

    # Study parameters
    n_observations: int = 100  # keep: 100
    n_features: int = 20  # keep: 20
    n_annotations_per_features: int = 2  # keep: 2

    # Model parameters
    h2: float = 0.9

    # General parameters
    model_format: Literal["dense", "sparse"] = "dense"
    density: float = 0.4  # only used if model_format is "sparse"

    path_script: Path = Path(__file__).parent
    path_inputs: Path = path_script / "inputs_brainiac"
    path_reference: Path = path_inputs / "reference"

    path_inputs.mkdir(parents=True, exist_ok=True)
    path_reference.mkdir(parents=True, exist_ok=True)

    save_precision_matrices: bool = False

    # 1. Generate random Z matrix
    z = np.random.rand(n_features, n_annotations_per_features)

    # 2. Sample alpha(s) from N(0, 1)
    alpha = np.random.normal(0, 1, (n_annotations_per_features, 1))
    print("alpha: ", alpha.flatten())

    # 3. Construct reference hyperparameters vector
    theta = np.concatenate(([h2], alpha.flatten()))

    # 4. Construct reference prior precision matrix
    normalized_exp_Z_alpha = np.exp(z @ alpha) / np.sum(np.exp(z @ alpha))

    h2_phi = h2 * normalized_exp_Z_alpha.flatten()
    Q_prior = sp.diags(1 / h2_phi)

    # 5. Generate random projection matrix "a"
    if model_format == "dense":
        a = np.random.rand(n_observations, n_features)
    elif model_format == "sparse":
        a = sp.random(
            n_observations, n_features, density=density, format="csc", dtype=np.float64
        )

    # 6. Construct reference beta vector
    var = 1 / Q_prior.diagonal()
    beta = np.random.normal(0, np.sqrt(var), n_features)

    # 7. Generate observation vector: sample full model: Y = a beta + epsilon
    if model_format == "dense":
        y = a @ beta
    elif model_format == "sparse":
        y = a.dot(beta)
    y += np.random.normal(0, np.sqrt(1 - h2), n_observations)

    # 8. Construct reference conditional precision matrix
    if model_format == "dense":
        Q_conditional = Q_prior.toarray() + 1 / (1 - h2) * (a.T @ a)
    elif model_format == "sparse":
        Q_conditional = Q_prior + 1 / (1 - h2) * (a.T @ a).tocsc()

    # Print Summary
    print("Generated BRAINIAC synthetic dataset with the following parameters:")
    print(f" - Number of observations: {n_observations}")
    print(f" - Number of features (latent variables): {n_features}")
    print(f" - Number of annotations per feature: {n_annotations_per_features}")
    print(f" - Model format: {model_format}")
    if model_format == "sparse":
        print(f" - Projection matrix density: {density}")
    print(f" - h2: {h2}")
    print()
    print("Saved the following files:")
    print(f" - BRAINIAC inputs in {path_inputs.resolve()}")
    print(f" - BRAINIAC reference outputs in {path_reference.resolve()}")

    # Save BRAINIAC inputs
    np.save(path_inputs / "z.npy", z)
    if model_format == "dense":
        np.save(path_inputs / "a.npy", a)
    elif model_format == "sparse":
        sp.save_npz(path_inputs / "a.npz", a)
    np.save(path_script / "y.npy", y)
    model_params = {
        "n_observations": n_observations,
        "n_features": n_features,
        "n_annotations_per_features": n_annotations_per_features,
        "h2": h2,
    }
    np.save(path_inputs / "model_params.npy", model_params)

    # Save BRAINIAC references
    np.save(path_reference / "theta.npy", theta)
    np.save(path_reference / "beta.npy", beta)

    if save_precision_matrices:
        sp.save_npz(path_reference / "Q_prior.npz", Q_prior)
        if model_format == "dense":
            np.save(path_reference / "Q_conditional.npy", Q_conditional)
        elif model_format == "sparse":
            sp.save_npz(path_reference / "Q_conditional.npz", Q_conditional)
