import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

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
    
    def plot_distribution(self, save_path: str | None = None):
        """Visualize probability distribution and payoff function"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot probability distribution
        ax1.bar(
            [(a + b)/2 for a, b in zip(self.price_bins[:-1], self.price_bins[1:])],
            self.probabilities,
            width=np.diff(self.price_bins),
            alpha=0.5
        )
        ax1.set_title("Probability Distribution")
        ax1.set_xlabel("Price")
        ax1.set_ylabel("Probability")
        
        # Plot payoff function
        S = np.linspace(min(self.price_bins), max(self.price_bins), 1000)
        payoffs = [self.get_payoff_function()(x) for x in S]
        ax2.plot(S, payoffs, 'r-', lw=2)
        ax2.set_title("Derived Payoff Function")
        ax2.set_xlabel("Price")
        ax2.set_ylabel("Payoff")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

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
    converter.plot_distribution(save_path="gaussian_mixture_payoff.png")
    
    # Test some values
    for price in [90, 95, 100, 105, 110]:
        print(f"Payoff at {price:.2f}%: {converter.get_payoff_function()(price):.4f}")

if __name__ == "__main__":
    example_usage()
