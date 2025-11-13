"""Loss functions for segmentation and classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for segmentation.

    Default: 0.5 * Dice + 0.5 * BCE
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1.0):
        """
        Args:
            dice_weight: Weight for Dice loss component
            bce_weight: Weight for BCE loss component
            smooth: Smoothing factor for Dice
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: Predicted logits (B, 1, H, W)
            targets: Ground truth masks (B, 1, H, W) with values 0 or 1

        Returns:
            Combined loss
        """
        # BCE loss
        bce = self.bce(logits, targets.float())

        # Dice loss
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2, 3)) + self.smooth
        den = (probs + targets).sum(dim=(2, 3)) + self.smooth
        dice = 1 - (num / den).mean()

        # Combine
        return self.dice_weight * dice + self.bce_weight * bce


class TverskyLoss(nn.Module):
    """Tversky loss for handling class imbalance in segmentation.

    α controls false negatives, β controls false positives.
    α=0.7, β=0.3 emphasizes recall (good for small lesions).
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        """
        Args:
            alpha: Weight for false negatives
            beta: Weight for false positives
            smooth: Smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: Predicted logits (B, 1, H, W)
            targets: Ground truth masks (B, 1, H, W)

        Returns:
            Tversky loss
        """
        # TODO: Implement Tversky loss
        raise NotImplementedError


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss for even stronger emphasis on hard examples."""

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.0,
        smooth: float = 1.0,
    ):
        """
        Args:
            alpha: Weight for false negatives
            beta: Weight for false positives
            gamma: Focal parameter (higher = more focus on hard examples)
            smooth: Smoothing factor
        """
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta, smooth)
        self.gamma = gamma

    def forward(self, logits, targets):
        """Focal Tversky: (1 - Tversky_index)^gamma"""
        # TODO: Implement
        raise NotImplementedError


class BoundaryLoss(nn.Module):
    """Boundary loss for improved contour accuracy.

    Computes distance transform of boundaries and penalizes errors.
    """

    def __init__(self, theta0: float = 3, theta: float = 5):
        """
        Args:
            theta0: Distance threshold
            theta: Smoothing parameter
        """
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, logits, targets):
        """
        Args:
            logits: Predicted logits (B, 1, H, W)
            targets: Ground truth masks (B, 1, H, W)

        Returns:
            Boundary loss
        """
        # TODO: Implement boundary loss using distance transforms
        raise NotImplementedError


class FocalLoss(nn.Module):
    """Focal loss for multi-class classification with class imbalance."""

    def __init__(
        self,
        alpha=None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha: Class weights (None or list/tensor of length num_classes)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: Predicted logits (B, C) for C classes
            targets: Ground truth labels (B,) as long

        Returns:
            Focal loss
        """
        # FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # Probability of true class
        focal_weight = (1 - p_t) ** self.gamma

        # Apply class weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha = torch.tensor(self.alpha, device=logits.device)
            else:
                alpha = self.alpha
            alpha_t = alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
