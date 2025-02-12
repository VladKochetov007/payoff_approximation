import numpy as np
from scipy.optimize import minimize
from typing import Callable, Tuple
import matplotlib
matplotlib.use('Agg')  # Добавить в начало файла
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

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
    
    # Create figure with improved styling
    plt.figure(figsize=(14, 7))
    plt.plot(S_test, target_values, 'k-', label='Target Payout', lw=3)
    plt.plot(S_test, approx_l2, '--', label='L2 Regularization Approximation', lw=2)
    plt.plot(S_test, approx_l1, '-.', label='L1 Regularization Approximation', lw=2)
    plt.title("Regularization Methods Comparison", fontsize=9)
    plt.xlabel("Underlying Asset Price at Maturity", fontsize=8)
    plt.ylabel("Payout", fontsize=8)
    plt.legend(fontsize=8, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=7)
    
    # Save data for pgfplots
    with open("../latex/regularization_comparison.dat", "w") as f:
        f.write("# S target l2 l1\n")
        for s, t, l2, l1 in zip(S_test, target_values, approx_l2, approx_l1):
            f.write(f"{s:.6f} {t:.6f} {l2:.6f} {l1:.6f}\n")
            
    # Generate pgfplots tex file
    with open("../latex/regularization_comparison.tex", "w") as f:
        f.write(r"""\begin{tikzpicture}
\begin{axis}[
    width=0.9\textwidth,
    height=6cm,
    grid=both,
    grid style={line width=.1pt, draw=gray!10},
    major grid style={line width=.2pt,draw=gray!50},
    xlabel style={font=\tiny},
    ylabel style={font=\tiny},
    tick label style={font=\tiny},
    title style={font=\small},
    legend style={font=\tiny, at={(0.02,0.98)}, anchor=north west},
    xlabel={Underlying Asset Price at Maturity},
    ylabel={Payout},
    title={Regularization Methods Comparison}
]

\addplot[thick, black] table[x index=0,y index=1] {regularization_comparison.dat};
\addlegendentry{Target Payout}

\addplot[thick, dashed, red] table[x index=0,y index=2] {regularization_comparison.dat};
\addlegendentry{L2 Regularization Approximation}

\addplot[thick, dashdotted, blue] table[x index=0,y index=3] {regularization_comparison.dat};
\addlegendentry{L1 Regularization Approximation}

\end{axis}
\end{tikzpicture}
""")
    
    # Also save PNG for quick preview
    plt.savefig("../latex/regularization_comparison.png", dpi=300, bbox_inches='tight')
    
    # Print results summary
    print("L2 Weights:")
    for K, w_call, w_put in zip(strikes, weights_l2[::2], weights_l2[1::2]):
        print(f"Strike {K:3.0f}: Call={w_call:+.3f}, Put={w_put:+.3f}")
    
    print("\nL1 Weights:")
    for K, w_call, w_put in zip(strikes, weights_l1[::2], weights_l1[1::2]):
        print(f"Strike {K:3.0f}: Call={w_call:+.3f}, Put={w_put:+.3f}")

if __name__ == "__main__":
    example_usage()
