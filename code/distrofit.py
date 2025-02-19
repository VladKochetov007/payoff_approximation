import numpy as np
from typing import Callable
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DistributionToPayoff:
    """
    Converts probability distribution forecast to payoff function
    
    Args:
        price_bins (list[float]): List of price bin edges
        probabilities (list[float]): Probability for each bin
        payoff_scale (float): Scaling factor for payoff amounts
    """
    
    def __init__(
        self,
        price_bins: list[float],
        probabilities: list[float],
        payoff_scale: float = 100.0
    ):
        self.price_bins = price_bins
        self.probabilities = probabilities
        self.payoff_scale = payoff_scale
        
        if len(price_bins) != len(probabilities) + 1:
            raise ValueError("Length of price_bins must be one more than probabilities")
    
    def get_payoff_function(self) -> Callable[[float], float]:
        """Create piecewise constant payoff function"""
        
        # Ensure probabilities sum to 1
        self.probabilities = np.array(self.probabilities) / sum(self.probabilities)
        
        # Calculate payoff weights: inverse probability weighting
        weights = self.payoff_scale * (1 / (self.probabilities + 1e-8))
        
        # Create piecewise constant function
        bins = self.price_bins
        
        def payoff(S: float) -> float:
            for i, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
                if lower <= S < upper:
                    return weights[i]
            return 0.0
        
        return payoff
    
    def plot_distribution(self):
        """Visualize probability distribution and payoff function in browser"""
        
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1)
        
        # Plot probability distribution
        bin_centers = [(a + b)/2 for a, b in zip(self.price_bins[:-1], self.price_bins[1:])]
        fig.add_trace(
            go.Bar(x=bin_centers, y=self.probabilities, name="Probability Distribution"),
            row=1, col=1
        )
        
        # Plot payoff function
        S = np.linspace(min(self.price_bins), max(self.price_bins), 1000)
        payoffs = [self.get_payoff_function()(x) for x in S]
        fig.add_trace(
            go.Scatter(x=S, y=payoffs, name="Payoff Function", line=dict(color='red', width=2)),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title_text="Distribution and Payoff Function",
            showlegend=True,
            height=800
        )
        
        fig.update_xaxes(title_text="Price", row=1, col=1)
        fig.update_xaxes(title_text="Price", row=2, col=1)
        fig.update_yaxes(title_text="Probability", row=1, col=1)
        fig.update_yaxes(title_text="Payoff", row=2, col=1)
        
        fig.show()

def example_usage():
    # Create example distribution (mixture of two Gaussians)
    x = np.linspace(80, 120, 41)
    dist1 = stats.norm.pdf(x, loc=95, scale=3)
    dist2 = stats.norm.pdf(x, loc=105, scale=4)
    probs = 0.4*dist1 + 0.6*dist2
    probs = probs / sum(probs)  # normalize
    
    # Convert to payoff function
    converter = DistributionToPayoff(x, probs)
    
    # Visualize
    converter.plot_distribution()
    
    # Test some values
    for price in [90, 95, 100, 105, 110]:
        print(f"Payoff at {price:.2f}%: {converter.get_payoff_function()(price):.4f}")

if __name__ == "__main__":
    example_usage()
