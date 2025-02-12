import numpy as np
from typing import Callable
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

    def plot_distribution(self):
        """Visualize probability distribution and payout function in browser"""
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        # Bin centers and widths
        centers = (self.bins[1:] + self.bins[:-1]) / 2
        
        # Plot probability distribution
        fig.add_trace(
            go.Bar(x=centers, y=self.probabilities, name="Probability Distribution"),
            row=1, col=1
        )
        
        # Plot payout function
        S = np.linspace(self.bins[0], self.bins[-1], 1000)
        payouts = [self.get_payout_function()(x) for x in S]
        fig.add_trace(
            go.Scatter(x=S, y=payouts, name="Payout Function", line=dict(color='red', width=2)),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title_text="Distribution and Payout Function",
            showlegend=True,
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Price Change (%)", row=2, col=1)
        fig.update_yaxes(title_text="Probability Density", row=1, col=1)
        fig.update_yaxes(title_text="Payout", row=2, col=1)
        
        fig.show()

# Example usage
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
    
    # Visualization (will open in browser)
    converter.plot_distribution()
    
    # Test function
    test_prices = [0.5, 1.0, 2.0, 3.0, 4.0]
    for price in test_prices:
        print(f"Payout at {price:.2f}%: {converter.get_payout_function()(price):.4f}")
