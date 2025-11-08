"""Temperature scaling for model calibration."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


class ModelWithTemperature(nn.Module):
    """Wrapper to apply temperature scaling to a trained model.

    Temperature scaling rescales logits to improve calibration:
    p_calibrated = softmax(logits / T)
    """

    def __init__(self, model):
        """
        Args:
            model: Trained classification model
        """
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Initialize to 1.5

    def forward(self, x):
        """
        Args:
            x: Input tensor

        Returns:
            Logits divided by temperature
        """
        logits = self.model(x)
        return logits / self.temperature.clamp(min=1e-3)

    def fit_temperature(self, val_loader: DataLoader, device: str = "cuda"):
        """Fit temperature parameter on validation set using NLL loss.

        Args:
            val_loader: Validation data loader
            device: Device to run on
        """
        # TODO: Implement
        # - Freeze model weights
        # - Optimize temperature using NLL loss on val set
        # - Use LBFGS optimizer (common for temperature scaling)
        raise NotImplementedError


def reliability_diagram(probs, targets, n_bins=10):
    """Generate reliability diagram data.

    Args:
        probs: Predicted probabilities (N,)
        targets: Ground truth labels (N,)
        n_bins: Number of bins

    Returns:
        tuple: (bin_confidences, bin_accuracies, bin_counts)
    """
    # TODO: Implement
    # - Bin predictions by confidence
    # - Compute average confidence and accuracy per bin
    # - Return for plotting
    raise NotImplementedError
