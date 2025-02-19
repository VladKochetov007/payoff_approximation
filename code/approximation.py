import numpy as np
from typing import Callable, Tuple
from sklearn.linear_model import Lasso
from scipy.optimize import minimize
import os


USE_CALL_PUT_PARITY = False
regularization = 0.1

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEX_DIR = os.path.join(BASE_DIR, "latex")

def ensure_dir(file_path):
    """Create directory if it doesn't exist"""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_basis_functions(strikes: list[float]) -> list[Callable[[float], float]]:
    """Generate call payoff functions and spot position using put-call parity"""
    basis = []
    for K in strikes:
        # Only call options needed (puts can be expressed via calls + spot)
        basis.append(lambda S, K=K: np.maximum(S - K, 0))
        if not USE_CALL_PUT_PARITY:
            # Put option payoff: max(K - S, 0) 
            basis.append(lambda S, K=K: np.maximum(K - S, 0))

    return basis

def weighted_error_objective(weights: np.ndarray, A: np.ndarray, b: np.ndarray, gamma: float) -> float:
    """
    Compute weighted error objective function
    
    Args:
        weights: Current solution vector
        A: Design matrix
        b: Target values
        gamma: Regularization parameter
        
    Returns:
        Weighted error value
    """
    pred = A @ weights
    error = np.abs(pred - b)
    weighted_error = np.sum(np.abs(b) * error)
    regularization = gamma * np.sum(np.abs(weights))
    return weighted_error + regularization

def approximate_payoff(
    target_payoff: Callable[[float], float],
    strikes: list[float],
    spot: float,
    regularization: float = 0.05,
    method: str = 'l2'
) -> Tuple[np.ndarray, float]:
    """
    Approximate target payoff using vanilla options
    
    Args:
        target_payoff: Function of S (spot price) to approximate
        strikes: Available strike prices
        spot: Current spot price
        regularization: Regularization parameter
        method: Method of regularization ('l2', 'l1', or 'weighted')
        
    Returns:
        Tuple (weights, lambda) where:
        - weights: Coefficients for options
        - lambda: Spot position coefficient
    """
    # Generate evaluation points around spot price
    S_values = np.linspace(0.5*spot, 1.5*spot, 100)
    
    # Create basis functions (calls/puts)
    basis = create_basis_functions(strikes)
    
    # Add spot position to the basis if using put-call parity
    if USE_CALL_PUT_PARITY:
        basis.append(lambda S: S)  # Spot position
    
    # Build design matrix
    A = np.array([[f(S) for f in basis] for S in S_values])
    b = np.array([target_payoff(S) for S in S_values])
    
    if method == 'l2':
        # Solve regularized least squares
        solution = np.linalg.lstsq(
            A.T @ A + regularization * np.eye(A.shape[1]),
            A.T @ b,
            rcond=None
        )[0]
    elif method == 'l1':
        # L1 regularization using Lasso
        model = Lasso(alpha=regularization, fit_intercept=False, max_iter=10000)
        model.fit(A, b)
        solution = model.coef_
    elif method == 'weighted':
        # Weighted error optimization
        x0 = np.zeros(A.shape[1])
        result = minimize(
            weighted_error_objective,
            x0,
            args=(A, b, regularization),
            method='SLSQP',
            options={'maxiter': 1000}
        )
        solution = result.x
    else:
        raise ValueError("Invalid method. Use 'l1', 'l2', or 'weighted'")
    
    if USE_CALL_PUT_PARITY:
        # Split solution into options weights and lambda
        return solution[:-1], solution[-1]
    else:
        # Return all weights and 0 for lambda since no spot position used
        return solution, 0.0

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def generate_comparison_table(
    target_payoff: Callable[[float], float],
    strikes: list[float],
    spot: float,
    regularization_values: list[float]
) -> None:
    """Generate performance comparison table for different regularization values"""
    # Generate evaluation points
    S_test = np.linspace(50, 150, 500)
    target_values = target_payoff(S_test)
    
    # Initialize results storage
    results = []
    
    for reg in regularization_values:
        # Calculate approximations for each method
        weights_l2, lambda_l2 = approximate_payoff(target_payoff, strikes, spot, reg, 'l2')
        weights_l1, lambda_l1 = approximate_payoff(target_payoff, strikes, spot, reg, 'l1')
        weights_weighted, lambda_weighted = approximate_payoff(target_payoff, strikes, spot, reg, 'weighted')
        
        # Calculate approximated values
        basis = create_basis_functions(strikes)
        approx_l2 = sum(w*f(S_test) for w, f in zip(weights_l2, basis))
        approx_l1 = sum(w*f(S_test) for w, f in zip(weights_l1, basis))
        approx_weighted = sum(w*f(S_test) for w, f in zip(weights_weighted, basis))
        
        if USE_CALL_PUT_PARITY:
            approx_l2 += lambda_l2*S_test
            approx_l1 += lambda_l1*S_test
            approx_weighted += lambda_weighted*S_test
        
        # Calculate MAPE for each method
        mape_l2 = calculate_mape(target_values, approx_l2)
        mape_l1 = calculate_mape(target_values, approx_l1)
        mape_weighted = calculate_mape(target_values, approx_weighted)
        
        results.append((reg, mape_l2, mape_l1, mape_weighted))
    
    # Generate LaTeX table
    method_comp_tex = os.path.join(LATEX_DIR, "method_comparison.tex")
    with open(method_comp_tex, "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Performance Comparison of Different Regularization Methods}
\label{tab:method_comparison}
\begin{tabular}{cccc}
\hline
$\gamma$ & L2 MAPE (\%) & L1 MAPE (\%) & Weighted MAPE (\%) \\
\hline
""")
        for reg, mape_l2, mape_l1, mape_weighted in results:
            f.write(f"{reg:.3f} & {mape_l2:.2f} & {mape_l1:.2f} & {mape_weighted:.2f} \\\\\n")
        
        f.write(r"""\hline
\end{tabular}
\end{table}
""")

def example_usage():
    """Demonstrate approximation of complex payoff structure"""
    # Define complex target payoff function
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
        ) + 15
    
    # Available strikes (limited set)
    strikes = [70, 80, 90, 100, 105, 110, 120, 130, 98]
    
    # Run approximation with tighter regularization
    weights_l2, lambda_l2 = approximate_payoff(target, strikes, spot=100, regularization=regularization, method='l2')
    weights_l1, lambda_l1 = approximate_payoff(target, strikes, spot=100, regularization=regularization, method='l1')
    weights_weighted, lambda_weighted = approximate_payoff(target, strikes, spot=100, regularization=regularization, method='weighted')
    
    # Generate comparison data
    S_test = np.linspace(50, 150, 500)
    target_values = target(S_test)
    
    # Calculate approximated payoff
    basis = create_basis_functions(strikes)
    approx_l2 = sum(w*f(S_test) for w, f in zip(weights_l2, basis))
    approx_l1 = sum(w*f(S_test) for w, f in zip(weights_l1, basis))
    approx_weighted = sum(w*f(S_test) for w, f in zip(weights_weighted, basis))
    
    # Add spot position if using put-call parity
    if USE_CALL_PUT_PARITY:
        approx_l2 += lambda_l2*S_test
        approx_l1 += lambda_l1*S_test
        approx_weighted += lambda_weighted*S_test

    # Ensure latex directory exists
    os.makedirs(LATEX_DIR, exist_ok=True)

    # Save data for L1/L2 comparison
    reg_comp_dat = os.path.join(LATEX_DIR, "regularization_comparison.dat")
    with open(reg_comp_dat, "w") as f:
        f.write("# S target l2 l1\n")
        for s, t, l2, l1 in zip(S_test, target_values, approx_l2, approx_l1):
            f.write(f"{s:.6f} {t:.6f} {l2:.6f} {l1:.6f}\n")
            
    # Save data for weighted method comparison
    weighted_dat = os.path.join(LATEX_DIR, "weighted_loss.dat")
    with open(weighted_dat, "w") as f:
        f.write("# S target weighted\n")
        for s, t, w in zip(S_test, target_values, approx_weighted):
            f.write(f"{s:.6f} {t:.6f} {w:.6f}\n")
            
    # Generate L1/L2 comparison plot
    reg_comp_tex = os.path.join(LATEX_DIR, "regularization_comparison.tex")
    with open(reg_comp_tex, "w") as f:
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
    ylabel={Payoff},
    title={L1 and L2 Regularization Comparison}
]

\addplot[thick, black] table[x index=0,y index=1] {regularization_comparison.dat};
\addlegendentry{Target Payoff}

\addplot[thick, dashed, red] table[x index=0,y index=2] {regularization_comparison.dat};
\addlegendentry{L2 Approximation ($\gamma="""+f"{regularization:.2f}"+r"""$)}

\addplot[thick, dashdotted, blue] table[x index=0,y index=3] {regularization_comparison.dat};
\addlegendentry{L1 Approximation ($\gamma="""+f"{regularization:.2f}"+r"""$)}

\end{axis}
\end{tikzpicture}
""")

    # Generate weighted method comparison plot
    weighted_tex = os.path.join(LATEX_DIR, "weighted_loss.tex")
    with open(weighted_tex, "w") as f:
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
    ylabel={Payoff},
    title={Weighted Error Method Comparison}
]

\addplot[thick, black] table[x index=0,y index=1] {weighted_loss.dat};
\addlegendentry{Target Payoff}

\addplot[thick, dotted, green!50!black] table[x index=0,y index=2] {weighted_loss.dat};
\addlegendentry{Weighted Error ($\gamma="""+f"{regularization:.2f}"+r"""$)}

\end{axis}
\end{tikzpicture}
""")

    # Print results summary
    print("L2 Weights:")
    for K, w_call in zip(strikes, weights_l2):
        print(f"Strike {K:3.0f}: Call={w_call:+.3f}")
    if USE_CALL_PUT_PARITY:
        print(f"Spot position: {lambda_l2:.3f}")
    
    print("\nL1 Weights:")
    for K, w_call in zip(strikes, weights_l1):
        print(f"Strike {K:3.0f}: Call={w_call:+.3f}")
    if USE_CALL_PUT_PARITY:
        print(f"Spot position: {lambda_l1:.3f}")
        
    print("\nWeighted Error Weights:")
    for K, w_call in zip(strikes, weights_weighted):
        print(f"Strike {K:3.0f}: Call={w_call:+.3f}")
    if USE_CALL_PUT_PARITY:
        print(f"Spot position: {lambda_weighted:.3f}")
        
    # Generate comparison table for different regularization values
    regularization_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    generate_comparison_table(target, strikes, spot=100, regularization_values=regularization_values)

if __name__ == "__main__":
    example_usage()
