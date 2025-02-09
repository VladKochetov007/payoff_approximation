import numpy as np
from scipy.optimize import minimize
from typing import Callable, Tuple
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import tikzplotlib  # Добавить в импорты

def create_basis_functions(strikes: list[float], spot: float) -> list[Callable[[float], float]]:
    """Generate call/put payoff functions for given strikes"""
    basis = []
    for K in strikes:
        # Call option payoff: max(S - K, 0)
        basis.append(lambda S, K=K: np.maximum(S - K, 0))
        # Put option payoff: max(K - S, 0) 
        basis.append(lambda S, K=K: np.maximum(K - S, 0))
    return basis

def approximate_payout(
    target_payout: Callable[[float], float],
    strikes: list[float],
    spot: float,
    regularization: float = 0.05,
    method: str = 'l2'
) -> Tuple[np.ndarray, float]:
    """
    Approximate target payout using vanilla options
    
    Args:
        target_payout: Function of S (spot price) to approximate
        strikes: Available strike prices
        spot: Current spot price
        regularization: L2 regularization parameter
        method: Method of regularization ('l2' or 'l1')
        
    Returns:
        Tuple (weights, lambda) where:
        - weights: Coefficients for options
        - lambda: Spot position coefficient
    """
    # Generate evaluation points around spot price
    S_values = np.linspace(0.5*spot, 1.5*spot, 100)
    
    # Create basis functions (calls/puts)
    basis = create_basis_functions(strikes, spot)
    
    # Add spot position (lambda*S) to the basis
    basis.append(lambda S: S)  # Lambda term
    
    # Build design matrix
    A = np.array([[f(S) for f in basis] for S in S_values])
    b = np.array([target_payout(S) for S in S_values])
    
    if method == 'l2':
        # Solve regularized least squares
        solution = np.linalg.lstsq(
            A.T @ A + regularization * np.eye(A.shape[1]),
            A.T @ b,
            rcond=None
        )[0]
    elif method == 'l1':
        # L1 регуляризация с использованием Lasso
        model = Lasso(alpha=regularization, fit_intercept=False, max_iter=10000)
        model.fit(A, b)
        solution = model.coef_
    else:
        raise ValueError("Invalid method. Use 'l1' or 'l2'")
    
    # Split solution into options weights and lambda
    return solution[:-1], solution[-1]

def example_usage():
    """Demonstrate approximation of complex payout structure"""
    # Define complex target payout function
    def target(S):
        return np.where(
            S < 80,
            0.5*(S - 70),
            np.where(S < 100,
                     np.sin((S - 80)/20 * np.pi) * 10 + 5,
                     np.where(S < 120,
                              0.8*(120 - S)**1.5,
                              0)
                    )
        )
    
    # Available strikes (limited set)
    strikes = [70, 80, 90, 100, 105, 110, 120, 130, 98]
    
    # Run approximation with tighter regularization
    weights_l2, lambda_l2 = approximate_payout(target, strikes, spot=100, regularization=0.2, method='l2')
    weights_l1, lambda_l1 = approximate_payout(target, strikes, spot=100, regularization=0.2, method='l1')
    
    # Generate comparison data
    S_test = np.linspace(50, 150, 500)
    target_values = target(S_test)
    
    # Calculate approximated payout
    basis = create_basis_functions(strikes, spot=100)
    approx_l2 = sum(w*f(S_test) for w, f in zip(weights_l2, basis)) + lambda_l2*S_test
    approx_l1 = sum(w*f(S_test) for w, f in zip(weights_l1, basis)) + lambda_l1*S_test
    
    # Create figure
    plt.figure(figsize=(14, 7))
    plt.plot(S_test, target_values, 'k-', label='Target Payout', lw=3)
    plt.plot(S_test, approx_l2, '--', label='L2 Regularization', lw=2)
    plt.plot(S_test, approx_l1, '-.', label='L1 Regularization', lw=2)
    plt.title("Regularization Methods Comparison", fontsize=14)
    plt.xlabel("Underlying Asset Price", fontsize=12)
    plt.ylabel("Payout", fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Сохраняем в TikZ
    tikzplotlib.save("regularization_comparison.tex",
                     axis_width="\\textwidth",
                     axis_height="8cm",
                     textsize=10)
    
    # Альтернативный вариант с настройкой стиля
    tikzplotlib.clean_figure()
    tikzplotlib.save("regularization_comparison.tex",
                     extra_axis_parameters={
                         "width=0.9\\textwidth",
                         "height=6cm",
                         "legend style={font=\\footnotesize}",
                         "label style={font=\\small}",
                         "tick label style={font=\\scriptsize}"
                     })
    
    # Print results summary
    print("L2 Weights:")
    for K, w_call, w_put in zip(strikes, weights_l2[::2], weights_l2[1::2]):
        print(f"Strike {K:3.0f}: Call={w_call:+.3f}, Put={w_put:+.3f}")
    
    print("\nL1 Weights:")
    for K, w_call, w_put in zip(strikes, weights_l1[::2], weights_l1[1::2]):
        print(f"Strike {K:3.0f}: Call={w_call:+.3f}, Put={w_put:+.3f}")

if __name__ == "__main__":
    example_usage()
