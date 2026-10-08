from scipy.stats import norm
import numpy as np

from dalia import NDArray, xp
from dalia.utils import get_host, get_device


def compute_transformed_quantiles(mean_internal, var_internal, percentiles, transform):
    """
    Compute quantiles for a transformed distribution
    
    Parameters:
    - original_dist_params: (mean, std) for the internal/transformed distribution
    - percentiles: array of probability values (0, 1)
    - transform: TransformationFunction object
    
    Returns:
    - quantiles in original scale
    
    Notes:
    Computing Quantiles for Transformed Distributions

    Idea:
    If X ~ f(x) and Y = φ(X), then find quantiles of Y:
    1. For a given probability p, find q_p such that P(Y ≤ q_p) = p
    2. This is equivalent to P(φ(X) ≤ q_p) = p
    3. We suppose φ is bijective and monotonely increasing. Then, if F_X is the CDF of X, we can write:
    F_Y(q_p) = P(Y ≤ q_p) = P(φ(X) ≤ q_p) = P(X ≤ φ⁻¹(q_p)) = F_X(φ⁻¹(q_p))

    4. So φ⁻¹(q_p) = F_X⁻¹(p), where F_X⁻¹ is the quantile function of X
    5. Therefore: q_p = φ(F_X⁻¹(p))
    """
    
    # Step 1: Compute quantiles in internal scale
    quantiles_np = get_host(percentiles)
    mean_internal_np = get_host(mean_internal)
    var_internal_np = get_host(var_internal)

    internal_quantiles = norm.ppf(quantiles_np, loc=mean_internal_np, scale=var_internal_np**0.5)

    # copy qunatiles to device
    internal_quantiles = get_device(internal_quantiles)
    
    # Step 2: Transform back to original scale
    # If φ: original → internal, then original quantiles = φ⁻¹(internal quantiles)
    original_quantiles = transform(internal_quantiles, direction='backward')
    
    return original_quantiles

def compute_transformed_pdf(mean_internal, var_internal, x_internal, transform):
    """
    Compute PDF of transformed distribution using change of variables
    
    If Y = φ(X), then f_Y(y) = f_X(φ⁻¹(y)) * |dφ⁻¹/dy|
    But we want f_X(x) where x is in original scale, so:
    f_X(x) = f_Y(φ(x)) * |dφ/dx|
    """ 
        
    # PDF in internal scale
    x_internal_np = get_host(x_internal)
    mean_internal_np = get_host(mean_internal)
    var_internal_np = get_host(var_internal)
    pdf_internal_np = norm.pdf(x_internal_np, loc=mean_internal_np, scale=var_internal_np**0.5)

    # copy pdf_internal to device
    pdf_internal = get_device(pdf_internal_np)

    # Jacobian: derivative of transformation
    # Ensure x_internal is treated as array for vectorized operations
    x_original = transform(x_internal, direction='backward')
    jacobian = xp.abs(transform(x_original, direction='forward_jacobian'))

    # PDF values in original scale
    pdf_original = pdf_internal * jacobian
        
    return x_original, pdf_original

# Automatic bound calculation based on 4 (default) standard deviations in internal scale
def compute_bounds(mean_internal, var_internal, transform, n_std=4):
    """
    Compute plotting bounds based on n standard deviations in internal scale
    """
    # Internal scale bounds (±n standard deviations)
    internal_lower = mean_internal - n_std * var_internal**0.5
    internal_upper = mean_internal + n_std * var_internal**0.5

    # Transform to original scale
    original_lower = transform(internal_lower, direction='backward')
    original_upper = transform(internal_upper, direction='backward')

    return (internal_lower, internal_upper), (original_lower, original_upper)

###################################### TEST ######################################
if __name__ == "__main__":
    """
    Test reparametrization functions using a dummy gamma prior hyperparameter class.
    
    This test demonstrates how the reparametrization functions work with 
    transformation functions that have forward, backward, and jacobian directions.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Dummy Gamma Prior Hyperparameter Class
    class DummyGammaPrior:
        """
        Dummy implementation of gamma prior rescaling for testing purposes.
        Implements log transformation: forward = log(x), backward = exp(x)
        """
        def rescale_hyperparameters_to_internal(self, theta, direction):
            """Log transformation between positive (external) and unconstrained (internal) space"""
            if direction == "forward":
                return xp.log(theta)  # theta -> log(theta)
            elif direction == "backward": 
                return xp.exp(theta)  # log(theta) -> theta
            elif direction == "forward_jacobian":
                return 1.0 / theta   # d(log(theta))/d(theta) = 1/theta
            elif direction == "backward_jacobian":
                return theta         # d(exp(theta))/d(theta) = exp(theta) = theta
            else:
                raise ValueError(f"Unknown direction: {direction}")
    
    print("=" * 80)
    print("Testing Reparametrization Functions with Dummy Gamma Prior")
    print("=" * 80)
    
    # Create dummy gamma prior instance
    gamma_prior = DummyGammaPrior()
    
    # Define transform function compatible with reparametrization functions
    def transform_func(x, direction):
        return gamma_prior.rescale_hyperparameters_to_internal(x, direction)
    
    # Test parameters (internal space: log-normal distribution)
    mean_internal = 0.5   # mean of log(theta)
    std_internal = 0.8    # std of log(theta)
    
    print(f"Internal distribution parameters:")
    print(f"  Mean (log scale): {mean_internal:.3f}")
    print(f"  Std (log scale):  {std_internal:.3f}")
    print()
    
    # Test 1: Transform validation
    print("1. Testing transformation consistency:")
    test_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    print("   Original -> Internal -> Original (round-trip test)")
    for theta in test_values:
        # Forward transformation
        log_theta = transform_func(theta, "forward")
        
        # Backward transformation  
        theta_recovered = transform_func(log_theta, "backward")
        
        # Check error
        error = abs(theta - theta_recovered)
        
        print(f"   {theta:5.1f} -> {log_theta:6.3f} -> {theta_recovered:6.3f}, "
              f"error = {error:.2e}")
    print()
    
    # Test 2: Jacobian validation using finite differences
    print("2. Testing Jacobian accuracy (forward direction):")
    
    test_theta_vals = [0.5, 1.0, 2.0, 3.0]
    eps = 1e-8
    
    print("   θ     | Analytical | Numerical  | Error")
    print("   ------|------------|------------|----------")
    
    for theta in test_theta_vals:
        # Analytical jacobian
        jac_analytical = transform_func(theta, "forward_jacobian")
        
        # Numerical jacobian using finite differences
        f_plus = transform_func(theta + eps, "forward")
        f_minus = transform_func(theta - eps, "forward")
        jac_numerical = (f_plus - f_minus) / (2 * eps)
        
        error = abs(jac_analytical - jac_numerical)
        
        print(f"   {theta:4.1f} | {jac_analytical:10.6f} | {jac_numerical:10.6f} | {error:.2e}")
    print()
    
    # Test 3: Quantile computation
    print("3. Testing quantile computation:")
    
    percentiles = xp.array([0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975])
    original_quantiles = compute_transformed_quantiles(
        mean_internal, std_internal**0.5, percentiles, transform_func
    )
    
    print("   Percentile | Quantile (original scale)")
    print("   -----------|-----------------------")
    for p, q in zip(percentiles, original_quantiles):
        print(f"   {p:8.2f}   | {q:18.6f}")
    print()
    
    # Test 4: PDF computation and validation
    print("4. Testing PDF computation:")
    
    # Compute bounds for plotting
    (internal_lower, internal_upper), (original_lower, original_upper) = compute_bounds(
        mean_internal, std_internal, transform_func, n_std=4
    )
    
    print(f"   Internal bounds: [{internal_lower:.3f}, {internal_upper:.3f}]")
    print(f"   Original bounds: [{original_lower:.3f}, {original_upper:.3f}]")
    
    # Test PDF at specific points
    test_x_original = xp.array([0.5, 1.0, 2.0, 3.0, 5.0])
    
    print("   x (orig) | PDF (orig) | log(x)   | PDF (int)")
    print("   ---------|------------|----------|----------")
    
    x_internal = transform_func(test_x_original, "forward")
    x_original, pdf_orig = compute_transformed_pdf(mean_internal, std_internal**2, x_internal, transform_func)
    pdf_internal = norm.pdf(x_internal, loc=mean_internal, scale=std_internal)
    
    for i in range(len(test_x_original)):
        print(f"   {test_x_original[i]:7.2f}  | {pdf_orig[i]:10.6f} | {x_internal[i]:8.3f} | {pdf_internal[i]:8.6f}")
    print()
    
    # Test 5: Analytical validation for log-normal distribution
    print("5. Analytical validation (log-normal distribution):")
    
    # For log-normal distribution, we can compute analytical moments
    mu = mean_internal
    sigma = std_internal
    
    # Analytical log-normal statistics
    analytical_mean = xp.exp(mu + sigma**2/2)
    analytical_var = (xp.exp(sigma**2) - 1) * xp.exp(2*mu + sigma**2)
    analytical_std = xp.sqrt(analytical_var)
    
    # Numerical verification using quantiles
    # Mean ≈ 50th percentile for log-normal (approximately)
    median_quantile = compute_transformed_quantiles(
        mean_internal, std_internal**0.5, xp.array([0.5]), transform_func
    )[0]
    
    print(f"   Analytical mean: {analytical_mean:.6f}")
    print(f"   Analytical std:  {analytical_std:.6f}")
    print(f"   Median quantile: {median_quantile:.6f}")
    print(f"   Mean/Median ratio: {analytical_mean/median_quantile:.6f}")
    print("   (Should be > 1 for log-normal due to skewness)")
    print()
    
    # Test 6: PDF integration check (numerical verification)
    print("6. PDF integration check:")
    
    # Create fine grid for integration -> need to start in original scale for dx to be equidistant
    x_original = xp.linspace(original_lower, original_upper, 1000)
    x_internal = transform_func(x_original, "forward")
    x_original, pdf_values = compute_transformed_pdf(mean_internal, std_internal**2, x_internal, transform_func)
        
    # Numerical integration using trapezoidal rule
    dx = x_original[1] - x_original[0]
    integral = np.trapezoid(pdf_values, dx=dx)
    
    print(f"   Numerical integral of PDF: {integral:.6f}")
    print(f"   Should be close to 1.0, error: {abs(1.0 - integral):.6f}")
    print()
    
    # repeat with non-equidistant grid in original scale but equidistant in internal scale
    print("   Repeating PDF integration with equidistant grid in internal scale:")
    x_internal = np.linspace(internal_lower, internal_upper, 1000)
    x_original, pdf_values = compute_transformed_pdf(mean_internal, std_internal**2, x_internal, transform_func)
    
    integral = np.trapezoid(pdf_values, x=x_original)
    print(f"   Numerical integral of PDF: {integral:.6f}")
    print(f"   Should be close to 1.0, error: {abs(1.0 - integral):.6f}")
    print()
    
    # Test 7: Bounds computation for different n_std values
    print("7. Testing bounds computation:")
    
    for n_std in [1, 2, 3, 4]:
        (int_lower, int_upper), (orig_lower, orig_upper) = compute_bounds(
            mean_internal, std_internal, transform_func, n_std=n_std
        )
        
        print(f"   {n_std}σ bounds:")
        print(f"     Internal: [{int_lower:7.3f}, {int_upper:7.3f}]")
        print(f"     Original: [{orig_lower:7.3f}, {orig_upper:7.3f}]")
    print()
    
    # Test 8: Plotting PDFs in both scales
    print("8. Plotting PDFs in internal and original scales:")
        
    # Plot 1: PDF in internal scale (log-scale, normal distribution)
    x_internal = xp.linspace(internal_lower, internal_upper, 500)
    pdf_internal = norm.pdf(x_internal, loc=mean_internal, scale=std_internal)
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(x_internal, pdf_internal, 'b-', linewidth=2, label='Internal PDF')
    internal_quantiles = norm.ppf(percentiles, loc=mean_internal, scale=std_internal)
    for i, (p, q) in enumerate(zip(percentiles, internal_quantiles)):
        color = 'red' if p in [0.025, 0.975] else 'orange'
        ax1.axvline(q, color=color, linestyle='--', alpha=0.7)
        if i % 2 == 0:
            ax1.text(q, max(pdf_internal)*0.8, f'{p*100:.1f}%', rotation=90, ha='right', va='top')

    ax1.set_title(f'Internal Scale: N(mean = {mean_internal}, std = {std_internal:.3f})')
    ax1.set_xlabel('θ (internal)')
    ax1.set_ylabel('PDF')
    ax1.set_xlim(x_internal[0], x_internal[-1])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Original distribution with quantiles
    x_original = np.linspace(int_lower, int_upper, 1000)
    x_original, pdf_original = compute_transformed_pdf(mean_internal, std_internal**2, x_internal, transform_func)

    ax2.plot(x_original, pdf_original, 'g-', linewidth=2, label='Original PDF')
    for i, (p, q) in enumerate(zip(percentiles, original_quantiles)):
        color = 'red' if p in [0.025, 0.975] else 'orange'
        ax2.axvline(q, color=color, linestyle='--', alpha=0.7)
        if i % 2 == 0:
            ax2.text(q, max(pdf_original)*0.8, f'{p*100:.1f}%', rotation=90, ha='right', va='top')

    ax2.set_title('Original Scale: Log-Normal Distribution')
    ax2.set_xlabel('θ_outer (original)')
    ax2.set_ylabel('PDF')
    ax2.set_xlim(orig_lower, 15)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.show()
            
    print("=" * 80)