import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

class DistributionToPayout:
    """
    Converts probability distribution forecast to payout function
    
    Parameters:
        bins (np.ndarray): Array of bin edges (N+1 elements)
        probabilities (np.ndarray): Probability for each bin (N elements)
        payout_scale (float): Scaling factor for payout amounts
    """
    def __init__(
        self,
        bins: np.ndarray,
        probabilities: np.ndarray,
        payout_scale: float = 100.0
    ):
        self.bins = bins
        self.probabilities = probabilities
        self.payout_scale = payout_scale
        
        # Validate inputs
        assert len(bins) == len(probabilities) + 1, "Invalid bins-probabilities shape"
        assert np.isclose(probabilities.sum(), 1.0), "Probabilities must sum to 1"
        
    def get_payout_function(self) -> Callable[[float], float]:
        """Create piecewise constant payout function"""
        # Calculate midpoints for each bin
        midpoints = (self.bins[1:] + self.bins[:-1]) / 2
        
        # Calculate payout weights: inverse probability weighting
        weights = self.payout_scale * (1 / (self.probabilities + 1e-8))
        
        # Normalize weights to [0, 1] range
        weights /= weights.max()
        
        def payout(S: float) -> float:
            # Find bin index for given price
            bin_idx = np.searchsorted(self.bins, S, side='right') - 1
            if 0 <= bin_idx < len(weights):
                return weights[bin_idx] * self.probabilities[bin_idx]
            return 0.0  # Out of range
            
        return payout

    def plot_distribution(self, save_path: str = None):
        """Visualize probability distribution and payout function"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Bin centers and widths
        centers = (self.bins[1:] + self.bins[:-1]) / 2
        widths = self.bins[1:] - self.bins[:-1]
        
        # Plot probability distribution
        ax1.bar(centers, self.probabilities, width=widths, alpha=0.7)
        ax1.set_title("Probability Distribution Forecast")
        ax1.set_ylabel("Probability Density")
        ax1.grid(True, alpha=0.3)
        
        # Plot payout function
        S = np.linspace(self.bins[0], self.bins[-1], 1000)
        payouts = [self.get_payout_function()(x) for x in S]
        ax2.plot(S, payouts, 'r-', lw=2)
        ax2.set_title("Derived Payout Function")
        ax2.set_xlabel("Price Change (%)")
        ax2.set_ylabel("Payout")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

# Updated usage example
if __name__ == "__main__":
    # Gaussian mixture parameters
    def gaussian_mixture(x, mu1=1.0, sigma1=0.5, mu2=3.0, sigma2=0.8, weight=0.3):
        return (weight * np.exp(-(x-mu1)**2/(2*sigma1**2)) / (sigma1*np.sqrt(2*np.pi)) 
                + (1-weight) * np.exp(-(x-mu2)**2/(2*sigma2**2)) / (sigma2*np.sqrt(2*np.pi)))

    # Create distribution based on Gaussian mixture
    price_bins = np.linspace(-0.1, 5.0, 101)
    bin_centers = (price_bins[1:] + price_bins[:-1])/2
    probs = gaussian_mixture(bin_centers)
    probs /= probs.sum()  # Normalize
    
    # Initialize converter
    converter = DistributionToPayout(price_bins, probs)
    
    # Visualization
    converter.plot_distribution(save_path="gaussian_mixture_payout.png")
    
    # Test function
    test_prices = [0.5, 1.0, 2.0, 3.0, 4.0]
    for price in test_prices:
        print(f"Payout at {price:.2f}%: {converter.get_payout_function()(price):.4f}")
