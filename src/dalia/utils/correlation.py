# Copyright 2024-2025 DALIA authors. All rights reserved.

import numpy as np
from dalia import xp
import matplotlib.pyplot as plt

# Import our quadrature functions
from dalia.utils.gaussian_quadrature import compute_variance_gauss_hermite
from dalia.utils.bivariate_gaussian_quadrature import compute_bivariate_expectation
# Import reparametrization functions
from dalia.utils.reparametrizations import (
    compute_transformed_quantiles, 
    compute_transformed_pdf, 
    compute_bounds
)

def compute_outer_covariance_matrix(mean_internal, cov_internal, transform_list, n_points=25):
    """
    Compute covariance matrix between all pairs of transformed parameters using bivariate quadrature.
    The results are to be considered with caution. Covariance matrices provide reliable information in 
    the Gaussian context, however, not necessarily in the transformed space.
    
    Parameters
    ----------
    mean_internal : ndarray
        Mean vector of internal distribution
    cov_internal : ndarray
        Covariance matrix of internal distribution
    transform_func : function
        Transformation function that takes (theta_vector, direction) and returns transformed vector
        This should be the model's rescale_hyperparameters_to_internal method
    n_points : int
        Number of quadrature points per dimension
        
    Returns
    -------
    ndarray
        Covariance matrix between transformed parameters (outer space)
    """
    
    n_dim = len(mean_internal)
    outer_cov_matrix = np.zeros((n_dim, n_dim))
        
    mean_outer = []
    marginal_vars = []
    
    print("Computing marginal means and variances for correlation calculations...")
    
    for i in range(n_dim):
        mu_i = mean_internal[i]
        var_i = cov_internal[i, i]
                
        # Compute marginal statistics
        result = compute_variance_gauss_hermite(mu_i, var_i, transform_list[i], n_points)
        mean_outer.append(result['mean'])
        marginal_vars.append(result['variance'])
        
        outer_cov_matrix[i, i] = result['variance']

    print("Computing pairwise covariances...")
    
    # Compute pairwise covariances
    for i in range(n_dim):
        for j in range(n_dim):
            if i < j:  # Only compute upper triangle, then symmetrize, i == j already filled
                print(f"  Computing Cov(X_{i+1}, X_{j+1})...", end=" ")
                
                # Extract marginal parameters
                mu_i, mu_j = mean_internal[i], mean_internal[j]
                var_i, var_j = cov_internal[i, i], cov_internal[j, j]
                cov_ij = cov_internal[i, j]
                
                # Compute correlation coefficient in internal space
                rho_internal = cov_ij / np.sqrt(var_i * var_j) 
                
                # Create transformation functions for these parameters
                transform_func_i = transform_list[i]
                transform_func_j = transform_list[j]

                # Standardize the variables for bivariate quadrature
                def standardized_func_i(z):
                    x_internal = mu_i + np.sqrt(var_i) * z
                    return transform_func_i(x_internal, "backward")
                
                def standardized_func_j(z):
                    x_internal = mu_j + np.sqrt(var_j) * z
                    return transform_func_j(x_internal, "backward")
                
                # Compute E[f_i(Z_i) * f_j(Z_j)] using bivariate quadrature
                cross_moment = compute_bivariate_expectation(
                    standardized_func_i, standardized_func_j, 
                    rho=rho_internal, n_points=n_points
                )
                                
                # Covariance: Cov(X,Y) = E[XY] - E[X]E[Y]
                covariance_outer = cross_moment - mean_outer[i] * mean_outer[j]
                
                # Store covariance directly
                outer_cov_matrix[i, j] = covariance_outer
                outer_cov_matrix[j, i] = covariance_outer  # Symmetric
                
                print(f"{covariance_outer:.6f}")
        
    return outer_cov_matrix


if __name__ == "__main__":

    """
    Main test function for multivariate transformations.
    """    

    # Dummy Prior Hyperparameter Class with Log Transform
    class Prior_LogTransform:
        """
        Dummy implementation of gamma prior rescaling for testing purposes.
        Implements log transformation: forward = log(x), backward = exp(x)
        """
        def rescale_hyperparameters_to_internal(self, theta, direction):
            """Log transformation between positive (external) and unconstrained (internal) space"""
            if direction == "forward":
                return np.log(theta)  # theta -> log(theta)
            elif direction == "backward": 
                return np.exp(theta)  # log(theta) -> theta
            elif direction == "forward_jacobian":
                return 1.0 / theta   # d(log(theta))/d(theta) = 1/theta
            elif direction == "backward_jacobian":
                return theta         # d(exp(theta))/d(theta) = exp(theta) = theta
            else:
                raise ValueError(f"Unknown direction: {direction}")
    
    class Prior_LogisticTransform:
        """
        Dummy implementation of beta prior rescaling for testing purposes.
        Implements logistic transformation: forward = logit(x), backward = sigmoid(x)
        """     
        
        def rescale_hyperparameters_to_internal(self, theta, direction):
            """Logistic transformation between (0,1) and unconstrained space"""
            if direction == "forward":
                return np.log(theta / (1 - theta))  # logit(theta)
            elif direction == "backward": 
                return 1 / (1 + np.exp(-theta))      # sigmoid(theta)
            elif direction == "forward_jacobian":
                return 1.0 / (theta * (1 - theta))   # d(logit(theta))/d(theta)
            elif direction == "backward_jacobian":
                sig = 1 / (1 + np.exp(-theta))
                return sig * (1 - sig)               # d(sigmoid(theta))/d(theta)
            else:
                raise ValueError(f"Unknown direction: {direction}")
            
    class Prior_IdentityTransform:
        """
        Dummy implementation of identity prior rescaling for testing purposes.
        Implements identity transformation: forward = x, backward = x
        """
        def rescale_hyperparameters_to_internal(self, theta, direction):
            """Identity transformation (no change)"""
            if direction in ["forward", "backward"]:
                return theta
            elif direction in ["forward_jacobian", "backward_jacobian"]:
                return 1.0
            else:
                raise ValueError(f"Unknown direction: {direction}")

    def generate_random_covariance_matrix(n_dim=3, condition_number=10.0, random_seed=42):
        """
        Generate a random positive definite covariance matrix.
        
        Parameters
        ----------
        n_dim : int
            Dimension of the covariance matrix
        condition_number : float
            Maximum condition number (controls how ill-conditioned the matrix can be)
        random_seed : int
            Random seed for reproducibility
            
        Returns
        -------
        ndarray
            Random positive definite covariance matrix
        """
        np.random.seed(random_seed)
        
        # Generate random eigenvalues between 1/condition_number and 1
        eigenvals = np.random.uniform(1.0/condition_number, 1.0, n_dim)
        eigenvals = np.sort(eigenvals)[::-1]  # Sort in descending order
        
        # Generate random orthogonal matrix (eigenvectors)
        Q, _ = np.linalg.qr(np.random.randn(n_dim, n_dim))
        
        # Construct covariance matrix: Σ = Q * diag(eigenvals) * Q^T
        cov_matrix = Q @ np.diag(eigenvals) @ Q.T
        
        return cov_matrix
    
    print("=" * 90)
    print("MULTIVARIATE TRANSFORMATION TEST")
    print("Testing 3D Gaussian → Transformed Space using Gaussian Quadrature")
    print("=" * 90)
    
    # Set parameters
    n_dim = 3
    n_quad_points = 30
    
    # Step 1: Generate random mean vector and covariance matrix
    print("1. Generating Random 3D Gaussian Distribution")
    print("-" * 50)
    
    np.random.seed(42)  # For reproducibility
    mean_internal = np.random.uniform(-1, 1, n_dim)
    cov_internal = generate_random_covariance_matrix(n_dim, condition_number=5.0)
    
    print("Internal (Gaussian) Distribution Parameters:")
    print(f"Mean vector: {mean_internal}")
    print("Covariance matrix:")
    print(cov_internal)
    print(f"Condition number: {np.linalg.cond(cov_internal):.2f}")
    print()
    
    # Step 2: Create transformation functions  
    print("2. Setting up Transformation Functions")
    print("-" * 50)

    prior_log_transform = Prior_LogTransform()

    def log_transform_func(x, direction):
        return prior_log_transform.rescale_hyperparameters_to_internal(x, direction)

    prior_logistic_transform = Prior_LogisticTransform()
    def logistic_transform_func(x, direction):
        return prior_logistic_transform.rescale_hyperparameters_to_internal(x, direction)
    
    prior_identity_transform = Prior_IdentityTransform()
    def identity_transform_func(x, direction):
        return prior_identity_transform.rescale_hyperparameters_to_internal(x, direction)
    
    # Apply different transforms to each dimension (directly assign transforms to parameters)
    transforms = [
        log_transform_func,  # Parameter 1: Log transform
        logistic_transform_func,  # Parameter 2: Logistic transform
        identity_transform_func  # Parameter 3: Identity transform
    ]
    
    # Step 3: Compute marginal statistics in outer space
    print("3. Computing Marginal Statistics in Outer Space")
    print("-" * 50)
    
    marginal_stats = []
    
    for i in range(n_dim):
        # Extract marginal parameters
        mean_i = mean_internal[i]
        var_i = cov_internal[i, i]
        
        # Compute marginal statistics
        stats = compute_variance_gauss_hermite(
            mean_i, var_i, transforms[i], n_quad_points
        )
        
        marginal_stats.append(stats)
        
        print(f"Parameter {i+1} ({stats['transform_name']}):")
        print(f"  Internal: μ = {mean_i:.4f}, σ² = {var_i:.4f}")
        print(f"  Outer:    μ = {stats['mean']:.4f}, σ² = {stats['variance']:.4f}, σ = {stats['std']:.4f}")
        print()
    
    # Step 4: Compute covariance matrix in outer space
    print("4. Computing Covariance Matrix in Outer Space")
    print("-" * 50)
    
    # Internal covariance matrix (for reference)
    internal_cov_matrix = cov_internal.copy()
    
    print("Internal (Gaussian) covariance matrix:")
    print(internal_cov_matrix)
    print()
    
    # Create a mock transformation function that applies different transforms to each parameter
    def mock_vectorized_transform(theta_vec, direction):
        """Mock transformation function that applies different transforms to each parameter."""
        result = theta_vec.copy()
        for i, transform_func in enumerate(transforms):
            if i < len(result):
                result[i] = transform_func(theta_vec[i], direction)
        return result
    
    # Compute outer covariance matrix using the new function
    outer_cov_matrix = compute_outer_covariance_matrix(
        mean_internal, cov_internal, mock_vectorized_transform, n_quad_points
    )
    
    print("\nOuter (Transformed) covariance matrix:")
    print(outer_cov_matrix)
    print()
    
    # Convert to correlation matrices for comparison
    # Internal correlations (for reference)
    internal_corr_matrix = np.zeros((n_dim, n_dim))
    for i in range(n_dim):
        for j in range(n_dim):
            if i == j:
                internal_corr_matrix[i, j] = 1.0
            else:
                internal_corr_matrix[i, j] = (cov_internal[i, j] / 
                                             np.sqrt(cov_internal[i, i] * cov_internal[j, j]))
    
    # Outer correlations (derived from covariance matrix)
    outer_corr_matrix = np.zeros((n_dim, n_dim))
    for i in range(n_dim):
        for j in range(n_dim):
            if i == j:
                outer_corr_matrix[i, j] = 1.0
            else:
                outer_corr_matrix[i, j] = (outer_cov_matrix[i, j] / 
                                          np.sqrt(outer_cov_matrix[i, i] * outer_cov_matrix[j, j]))
    
    print("Derived outer correlation matrix:")
    print(outer_corr_matrix)
    print()
    
    # Step 5: Compare transformations
    print("5. Transformation Effects Analysis")
    print("-" * 50)
    
    print("Comparison of Internal vs Outer Statistics:")
    print(f"{'Parameter':<12} {'Transform':<25} {'Mean Change':<12} {'Var Change':<12} {'Cov Change':<12}")
    print("-" * 85)
    
    for i in range(n_dim):
        mean_change = abs(marginal_stats[i]['mean'] - mean_internal[i])
        var_change = abs(marginal_stats[i]['variance'] - cov_internal[i, i])
        
        # Average covariance change for this parameter
        cov_changes = []
        for j in range(n_dim):
            if i != j:
                cov_changes.append(abs(outer_cov_matrix[i, j] - internal_cov_matrix[i, j]))
        avg_cov_change = np.mean(cov_changes) if cov_changes else 0.0
        
        transform_name = transforms[i].name.split(' ')[0]
        
        print(f"{i+1:<12} {transform_name:<25} {mean_change:<12.4f} {var_change:<12.4f} {avg_cov_change:<12.4f}")
    
    print()
    
    # Additional covariance matrix validation
    print("Covariance Matrix Validation:")
    print("-" * 50)
    
    # Check if outer covariance matrix is positive semidefinite
    eigenvals = np.linalg.eigvals(outer_cov_matrix)
    is_pos_def = np.all(eigenvals >= -1e-10)  # Allow small numerical errors
    
    print(f"Outer covariance matrix eigenvalues: {eigenvals}")
    print(f"Is positive semidefinite: {is_pos_def}")
    
    # Check symmetry
    is_symmetric = np.allclose(outer_cov_matrix, outer_cov_matrix.T)
    print(f"Is symmetric: {is_symmetric}")
    
    # Check diagonal elements (should be positive variances)
    diag_elements = np.diag(outer_cov_matrix)
    all_positive_vars = np.all(diag_elements > 0)
    print(f"All diagonal elements (variances) positive: {all_positive_vars}")
    print(f"Outer variances: {diag_elements}")
    
    print()
    
    # Step 6: Analytical validation for specific cases
    print("6. Analytical Validation")
    print("-" * 50)
    
    # For log transformation (Parameter 1), we can validate against log-normal theory
    if transforms[0].name.startswith("Log"):  # Log transform
        mu_1 = mean_internal[0]
        sigma2_1 = cov_internal[0, 0]
        
        # Analytical log-normal moments
        analytical_mean = np.exp(mu_1 + sigma2_1/2)
        analytical_var = (np.exp(sigma2_1) - 1) * np.exp(2*mu_1 + sigma2_1)
        
        numerical_mean = marginal_stats[0]['mean']
        numerical_var = marginal_stats[0]['variance']
        
        print("Log-normal validation (Parameter 1):")
        print(f"  Analytical mean: {analytical_mean:.6f}")
        print(f"  Numerical mean:  {numerical_mean:.6f}")
        print(f"  Relative error:  {abs(analytical_mean - numerical_mean)/analytical_mean:.2e}")
        print(f"  Analytical var:  {analytical_var:.6f}")
        print(f"  Numerical var:   {numerical_var:.6f}")
        print(f"  Relative error:  {abs(analytical_var - numerical_var)/analytical_var:.2e}")
        print()
        
        # Demonstrate reparametrization functions
        print("Reparametrization functions analysis:")
        
        # Create transform function for reparametrization utilities
        transform_func = transforms[0]  # Log transform
        def reparam_func(x, direction):
            return transform_func(x, direction)
        
        # Compute quantiles using reparametrization function
        percentiles = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
        quantiles = compute_transformed_quantiles(mu_1, sigma2_1, percentiles, reparam_func)
        
        print("  Quantiles in outer space:")
        for p, q in zip(percentiles, quantiles):
            print(f"    {p*100:4.1f}%: {q:.4f}")
        
        # Compute bounds using reparametrization function
        (int_lower, int_upper), (orig_lower, orig_upper) = compute_bounds(
            mu_1, sigma2_1, reparam_func, n_std=3
        )
        print(f"  3σ bounds: Internal [{int_lower:.3f}, {int_upper:.3f}] -> Outer [{orig_lower:.3f}, {orig_upper:.3f}]")
        
        # Compute PDF at a few points to demonstrate reparametrization
        test_points = [0.5, 1.0, 2.0, 5.0]
        print("  PDF values at test points:")
        for x_orig in test_points:
            x_int = reparam_func(x_orig, "forward")
            pdf_orig = compute_transformed_pdf(mu_1, sigma2_1, x_int, reparam_func)
            print(f"    x = {x_orig:.1f}: PDF = {pdf_orig:.6f}")
        print()
        print()
    
    # Step 7: Create visualization
    print("7. Generating Visualization")
    print("-" * 50)
    
    try:
        # Create single figure with 3 rows (one per parameter), 2 columns (internal, outer)
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # Plot marginal distributions for each parameter
        for i in range(n_dim):
            # Get marginal parameters
            mu_i = mean_internal[i]
            sigma_i = np.sqrt(cov_internal[i, i])
            transform_func = transforms[i]
            
            # Create transform function for reparametrization
            def param_transform_func(x, direction):
                return transform_func(x, direction)
            
            # === INTERNAL DISTRIBUTION PLOT ===
            ax_int = axes[i, 0]  # Row i, column 0 (internal)
            
            # Create well-spaced internal grid
            x_internal = np.linspace(mu_i - 4*sigma_i, mu_i + 4*sigma_i, 300)
            pdf_internal = (1/(sigma_i * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_internal - mu_i)/sigma_i)**2)
            
            ax_int.plot(x_internal, pdf_internal, 'b-', linewidth=3, label=f'N({mu_i:.2f}, {sigma_i:.2f}²)')
            
            # Add quantiles for internal distribution
            internal_percentiles = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
            from scipy.stats import norm
            internal_quantiles = norm.ppf(internal_percentiles, loc=mu_i, scale=sigma_i)
            
            colors_int = ['red', 'orange', 'green', 'orange', 'red']
            for p, q, color in zip(internal_percentiles, internal_quantiles, colors_int):
                pdf_val = (1/(sigma_i * np.sqrt(2*np.pi))) * np.exp(-0.5*((q - mu_i)/sigma_i)**2)
                ax_int.axvline(q, color=color, linestyle='--', alpha=0.7, linewidth=2)
                if p in [0.025, 0.5, 0.975]:  # Label key percentiles
                    ax_int.text(q, pdf_val * 1.05, f'{p:.3f}', rotation=90, ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Set internal plot properties
            ax_int.set_xlim(mu_i - 4*sigma_i, mu_i + 4*sigma_i)
            ax_int.set_ylim(0, max(pdf_internal) * 1.15)
            ax_int.set_xlabel(f'Parameter {i+1} (Internal Scale)', fontsize=12)
            ax_int.set_ylabel('PDF', fontsize=12)
            ax_int.set_title(f'Parameter {i+1}: Internal Distribution\n{transform_func.name}', fontsize=14, fontweight='bold')
            ax_int.legend(fontsize=11)
            ax_int.grid(True, alpha=0.3)
            
            # === OUTER DISTRIBUTION PLOT ===
            ax_out = axes[i, 1]  # Row i, column 1 (outer)
            
            try:
                # Compute bounds for outer distribution with more generous margins
                (int_lower, int_upper), (orig_lower, orig_upper) = compute_bounds(
                    mu_i, sigma_i**2, param_transform_func, n_std=4
                )
                
                # Add some margin to outer bounds for better visualization
                orig_range = orig_upper - orig_lower
                orig_margin = orig_range * 0.1
                orig_lower_plot = max(orig_lower - orig_margin, 1e-6) if orig_lower > 0 else orig_lower - orig_margin
                orig_upper_plot = orig_upper + orig_margin
                
                # Create fine grid in outer space
                x_outer = np.linspace(orig_lower_plot, orig_upper_plot, 300)
                
                # Filter out invalid values for certain transformations
                if "Logistic" in transforms[i].name:  # Logistic transform (0,1)
                    x_outer = x_outer[(x_outer > 0.001) & (x_outer < 0.999)]
                elif "Log" in transforms[i].name:  # Log transform (positive)
                    x_outer = x_outer[x_outer > 0.001]
                # Identity transform needs no filtering - can handle all real values
                
                # Compute PDF in outer space
                pdf_outer = []
                for x in x_outer:
                    try:
                        x_int = param_transform_func(x, "forward")
                        pdf_val = compute_transformed_pdf(mu_i, sigma_i**2, x_int, param_transform_func)
                        pdf_outer.append(pdf_val)
                    except:
                        pdf_outer.append(0.0)
                
                pdf_outer = np.array(pdf_outer)
                
                # Plot outer distribution
                ax_out.plot(x_outer, pdf_outer, 'r-', linewidth=3, label=f'Transformed Distribution')
                
                # Add quantiles for outer distribution
                outer_quantiles = compute_transformed_quantiles(mu_i, sigma_i**2, internal_percentiles, param_transform_func)
                
                colors_out = ['red', 'orange', 'green', 'orange', 'red']
                for p, q, color in zip(internal_percentiles, outer_quantiles, colors_out):
                    if orig_lower_plot <= q <= orig_upper_plot:  # Only plot if within bounds
                        try:
                            x_int_q = param_transform_func(q, "forward")
                            pdf_val_q = compute_transformed_pdf(mu_i, sigma_i**2, x_int_q, param_transform_func)
                            ax_out.axvline(q, color=color, linestyle='--', alpha=0.7, linewidth=2)
                            if p in [0.025, 0.5, 0.975]:  # Label key percentiles
                                ax_out.text(q, pdf_val_q * 1.05, f'{p:.3f}', rotation=90, ha='center', va='bottom', fontsize=10, fontweight='bold')
                        except:
                            pass
                
                # Set outer plot properties with proper limits
                ax_out.set_xlim(orig_lower_plot, orig_upper_plot)
                if len(pdf_outer) > 0 and max(pdf_outer) > 0:
                    ax_out.set_ylim(0, max(pdf_outer) * 1.15)
                
                # Format x-axis nicely for different transformations
                if "Log" in transforms[i].name:  # Log transform
                    ax_out.set_xlabel(f'Parameter {i+1} (Outer Scale: exp)', fontsize=12)
                elif "Logistic" in transforms[i].name:  # Logistic transform  
                    ax_out.set_xlabel(f'Parameter {i+1} (Outer Scale: sigmoid)', fontsize=12)
                elif "Identity" in transforms[i].name:  # Identity transform
                    ax_out.set_xlabel(f'Parameter {i+1} (Outer Scale: identity)', fontsize=12)
                
                ax_out.set_ylabel('PDF', fontsize=12)
                ax_out.set_title(f'Parameter {i+1}: Outer Distribution\n{transform_func.name}', fontsize=14, fontweight='bold')
                ax_out.legend(fontsize=11)
                ax_out.grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"Warning: Could not plot outer distribution for parameter {i+1}: {e}")
                ax_out.text(0.5, 0.5, f'Error plotting\nparameter {i+1}', transform=ax_out.transAxes, ha='center', va='center', fontsize=12)
                ax_out.set_title(f'Parameter {i+1}: Error', fontsize=14)
        
        # Add column labels
        axes[0, 0].text(0.5, 1.15, 'Internal (Gaussian) Scale', transform=axes[0, 0].transAxes, 
                        ha='center', va='bottom', fontsize=16, fontweight='bold')
        axes[0, 1].text(0.5, 1.15, 'Outer (Transformed) Scale', transform=axes[0, 1].transAxes, 
                        ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        # Finalize figure
        fig.suptitle('Marginal Distributions: Internal vs Outer Scales', fontsize=18, fontweight='bold', y=0.98)
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)  # Make room for suptitle and column headers
        fig.savefig('marginal_distributions_comparison.png', dpi=300, bbox_inches='tight')
        
        # Show figure
        plt.show()
        
        print("✓ Marginal distributions comparison saved as 'marginal_distributions_comparison.png'")
        
    except Exception as e:
        print(f"⚠ Could not generate plots: {e}")
    
    print()
    
    # Step 8: Summary
    print("8. Summary")
    print("-" * 50)
        
    # Calculate total changes in both covariance and correlation structures
    total_cov_change = np.sum(np.abs(outer_cov_matrix - internal_cov_matrix)) / 2  # Divide by 2 due to symmetry
    total_corr_change = np.sum(np.abs(outer_corr_matrix - internal_corr_matrix)) / 2  # Divide by 2 due to symmetry
    print(f"✓ Total covariance structure change: {total_cov_change:.6f}")
    print(f"✓ Total correlation structure change: {total_corr_change:.4f}")
    
    print()
    print("=" * 90)
    print("MULTIVARIATE TRANSFORMATION TEST COMPLETED!")
    print("=" * 90)


if __name__ == "__main__":
    test_multivariate_transformation()