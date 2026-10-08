# Copyright 2024-2025 DALIA authors. All rights reserved.

from dalia import xp

from scipy.special import roots_hermite

def compute_variance_gauss_hermite(mean_internal, variance_internal, transform, n_points=20):
    """
    Compute variance of transformed distribution using Gauss-Hermite quadrature
    
    For a distribution Y where transform(Y) ~ N(μ, σ²), we want to compute:
    Var(Y) = E[Y²] - (E[Y])²
    
    Using Gauss-Hermite quadrature by transforming to standard normal form.
    
    Theory:
    If Z ~ N(0,1), then X = μ + σZ ~ N(μ, σ²)
    So Y = φ⁻¹(X) = φ⁻¹(μ + σZ)
    
    E[Y] = E[φ⁻¹(X)] = E[φ⁻¹(μ + σZ)] 
    E[Y²] = E[φ⁻¹(X)φ⁻¹(X)] =  E[φ⁻¹(μ + σZ) φ⁻¹(μ + σZ)] 
    """
    
    # Get Gauss-Hermite quadrature points and weights
    nodes, weights = roots_hermite(n_points)
    
    # copy nodes and weight to device
    nodes = xp.array(nodes)
    weights = xp.array(weights)
    
    # Transform nodes from Hermite polynomial roots to standard normal
    # Hermite nodes are for exp(-x²), we want exp(-x²/2)/√(2π)
    # So we scale by √2: z = √2 * node
    z_nodes = xp.sqrt(2) * nodes
    
    # Compute transformed values Y = φ⁻¹(μ + σZ) for each node
    internal_values = mean_internal + xp.sqrt(variance_internal) * z_nodes
    y_values = transform(internal_values, direction="backward")
    
    # Gauss-Hermite weights need to be adjusted for standard normal
    # Original: ∫ f(x) exp(-x²) dx ≈ Σ w_i f(x_i)
    # For standard normal: ∫ f(z) (1/√(2π)) exp(-z²/2) dz
    # After substitution x = z/√2: ∫ f(√2 x) (1/√π) exp(-x²) dx
    adjusted_weights = weights / xp.sqrt(xp.pi)
    
    # Compute first and second moments
    mean_y = xp.sum(adjusted_weights * y_values)
    second_moment_y = xp.sum(adjusted_weights * y_values**2)

    # Variance = E[Y²] - (E[Y])²
    variance_y = second_moment_y - mean_y**2
    
    return {
        'mean': mean_y,
        'second_moment': second_moment_y,
        'variance': variance_y,
        'std': xp.sqrt(variance_y)
    }
    
   
#################################################################################
#### just for testing purposes below ##########################################    
# Dummy classes to avoid circular imports during testing
class DummyConfig:
    def __init__(self, alpha=2.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

class DummyGammaPriorHyperparameters:
    def __init__(self, config):
        self.config = config
    
    def rescale_hyperparameters_to_internal(self, theta, direction):
        """Log transformation: external (positive) <-> internal (unconstrained)"""
        if direction == "forward":
            return xp.log(theta)
        elif direction == "backward":
            return xp.exp(theta)
        else:
            raise ValueError(f"Unknown direction: {direction}")


def test_gaussian_quadrature():
    """
    Testing Gaussian quadrature function.
    
    Tests multiple transformation functions and compares results with 
    analytical solutions where available.
    """
    
    print("=" * 80)
    print("GAUSSIAN QUADRATURE TESTS")
    print("=" * 80)
    
    # Test parameters
    test_tolerance = 1e-6
    n_quad_points = 20
    
    # Test 1: Identity transformation (should recover original normal moments)
    print("1. Testing Identity Transformation (X = Y)")
    print("-" * 50)
    
    def identity_transform(x, direction):
        return x  # No transformation
    
    mu = 2.0
    sigma2 = 1.5
    
    result = compute_variance_gauss_hermite(mu, sigma2, identity_transform, n_quad_points)
    
    # For identity, we should recover the original normal distribution moments
    expected_mean = mu
    expected_variance = sigma2
    
    print(f"  Expected mean: {expected_mean:.6f}, Got: {result['mean']:.6f}")
    print(f"  Expected var:  {expected_variance:.6f}, Got: {result['variance']:.6f}")
    
    mean_error = abs(result['mean'] - expected_mean)
    var_error = abs(result['variance'] - expected_variance)
    
    print(f"  Mean error: {mean_error:.2e}")
    print(f"  Var error:  {var_error:.2e}")
    
    assert mean_error < test_tolerance, f"Identity mean test failed: error = {mean_error}"
    assert var_error < test_tolerance, f"Identity variance test failed: error = {var_error}"
    print("  ✓ Identity transformation test PASSED")
    print()
    
    # Test 2: Log transformation (log-normal distribution)
    print("2. Testing Log Transformation (Y = exp(X))")
    print("-" * 50)
    
    def log_transform(x, direction):
        if direction == "forward":
            return xp.log(x)
        elif direction == "backward":
            return xp.exp(x)
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    mu = 0.5
    sigma2 = 0.25
    
    result = compute_variance_gauss_hermite(mu, sigma2, log_transform, n_quad_points)
    
    # Analytical moments for log-normal: if log(Y) ~ N(μ, σ²)
    expected_mean = xp.exp(mu + sigma2/2)
    expected_variance = (xp.exp(sigma2) - 1) * xp.exp(2*mu + sigma2)
    
    print(f"  Expected mean: {expected_mean:.6f}, Got: {result['mean']:.6f}")
    print(f"  Expected var:  {expected_variance:.6f}, Got: {result['variance']:.6f}")
    
    mean_error = abs(result['mean'] - expected_mean) / expected_mean
    var_error = abs(result['variance'] - expected_variance) / expected_variance
    
    print(f"  Relative mean error: {mean_error:.2e}")
    print(f"  Relative var error:  {var_error:.2e}")
    
    assert mean_error < 1e-4, f"Log-normal mean test failed: rel error = {mean_error}"
    assert var_error < 1e-4, f"Log-normal variance test failed: rel error = {var_error}"
    print("  ✓ Log transformation test PASSED")
    print()
    
    # Test 3: Linear transformation (Y = aX + b)
    print("3. Testing Linear Transformation (Y = aX + b)")
    print("-" * 50)
    
    a, b = 3.0, -1.5
    
    def linear_transform(x, direction):
        if direction == "forward":
            return (x - b) / a
        elif direction == "backward":
            return a * x + b
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    mu = 1.0
    sigma2 = 0.8
    
    result = compute_variance_gauss_hermite(mu, sigma2, linear_transform, n_quad_points)
    
    # For Y = aX + b where X ~ N(μ, σ²): E[Y] = aμ + b, Var[Y] = a²σ²
    expected_mean = a * mu + b
    expected_variance = a**2 * sigma2
    
    print(f"  Transform: Y = {a}X + {b}")
    print(f"  Expected mean: {expected_mean:.6f}, Got: {result['mean']:.6f}")
    print(f"  Expected var:  {expected_variance:.6f}, Got: {result['variance']:.6f}")
    
    mean_error = abs(result['mean'] - expected_mean)
    var_error = abs(result['variance'] - expected_variance)
    
    print(f"  Mean error: {mean_error:.2e}")
    print(f"  Var error:  {var_error:.2e}")
    
    assert mean_error < test_tolerance, f"Linear mean test failed: error = {mean_error}"
    assert var_error < test_tolerance, f"Linear variance test failed: error = {var_error}"
    print("  ✓ Linear transformation test PASSED")
    print()

    # Test 4: Gamma prior rescaling (again log transformation)
    print("4. Testing Gamma Prior Rescaling")
    print("-" * 50)
    
    config = DummyConfig(alpha=2.0, beta=1.0)
    gamma_prior = DummyGammaPriorHyperparameters(config=config)
    
    def gamma_rescale(x, direction):
        return gamma_prior.rescale_hyperparameters_to_internal(x, direction)
    
    mu = 0.2
    sigma2 = 0.3
    
    result = compute_variance_gauss_hermite(mu, sigma2, gamma_rescale, n_quad_points)
    
    # This is the same as log-normal since rescale uses exp transformation
    expected_mean = xp.exp(mu + sigma2/2)
    expected_variance = (xp.exp(sigma2) - 1) * xp.exp(2*mu + sigma2)
    
    print(f"  Expected mean: {expected_mean:.6f}, Got: {result['mean']:.6f}")
    print(f"  Expected var:  {expected_variance:.6f}, Got: {result['variance']:.6f}")
    
    mean_error = abs(result['mean'] - expected_mean) / expected_mean
    var_error = abs(result['variance'] - expected_variance) / expected_variance
    
    print(f"  Relative mean error: {mean_error:.2e}")
    print(f"  Relative var error:  {var_error:.2e}")
    
    assert mean_error < 1e-4, f"Gamma rescale mean test failed: rel error = {mean_error}"
    assert var_error < 1e-4, f"Gamma rescale variance test failed: rel error = {var_error}"
    print("  ✓ Gamma prior rescaling test PASSED")
    print()

    # Test 5: Convergence with increasing quadrature points
    print("5. Testing Convergence with Quadrature Points")
    print("-" * 50)
    
    mu_conv = 0.1
    sigma2_conv = 0.4
    expected_mean_conv = xp.exp(mu_conv + sigma2_conv/2)
    
    n_points_list = [5, 10, 15, 20, 30, 50]
    errors_mean = []
    errors_var = []

    print("  n_points  |   Mean      |  Rel. Mean Error | Rel. Var. Error")
    print("  ----------|-------------|------------------|-----------------")
    
    for n in n_points_list:
        result = compute_variance_gauss_hermite(mu_conv, sigma2_conv, log_transform, n)
        rel_error_mean = abs(result['mean'] - expected_mean_conv) / expected_mean_conv
        errors_mean.append(rel_error_mean)

        print(f"  {n:8d}  | {result['mean']:10.6f}  | {rel_error_mean:5.3e}      ")

    # Check that errors generally decrease (allowing some numerical noise)
    improving = sum(errors_mean[i+1] < errors_mean[i] * 1.1 for i in range(len(errors_mean)-1))
    improvement_rate = improving / (len(errors_mean) - 1)

    print(f"  Improvement rate: {improvement_rate:.1%}")
    assert improvement_rate > 0.6, f"Convergence test failed: improvement rate = {improvement_rate}"
    print("  ✓ Convergence test PASSED")
    print()
    
    # Summary
    print("=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)


if __name__ == "__main__":
    test_gaussian_quadrature()