"""Metrics for segmentation and classification evaluation."""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


def dice_score(preds, targets, smooth=1.0):
    """Compute Dice coefficient.

    Args:
        preds: Binary predictions (B, 1, H, W) or (B, H, W)
        targets: Binary targets (B, 1, H, W) or (B, H, W)
        smooth: Smoothing factor

    Returns:
        Dice score (scalar)
    """
    # Dice = (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    preds = preds.flatten()
    targets = targets.flatten()

    intersection = (preds * targets).sum()
    dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

    return float(dice)


def iou_score(preds, targets, smooth=1.0):
    """Compute Intersection over Union (Jaccard index).

    Args:
        preds: Binary predictions
        targets: Binary targets
        smooth: Smoothing factor

    Returns:
        IoU score (scalar)
    """
    # IoU = (|X ∩ Y| + smooth) / (|X ∪ Y| + smooth)
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    preds = preds.flatten()
    targets = targets.flatten()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    return float(iou)


def boundary_iou(preds, targets, dilation=2):
    """Compute Boundary IoU (intersection over union of boundaries).

    Args:
        preds: Binary predictions (B, H, W)
        targets: Binary targets (B, H, W)
        dilation: Dilation size for boundary extraction

    Returns:
        Boundary IoU score
    """
    # TODO: Implement
    # - Extract boundaries using morphological operations
    # - Compute IoU of boundaries only
    raise NotImplementedError


def boundary_f1(preds, targets, dilation=2):
    """Compute Boundary F1 score.

    Args:
        preds: Binary predictions
        targets: Binary targets
        dilation: Dilation size

    Returns:
        Boundary F1 score
    """
    # TODO: Implement using boundary extraction + F1
    raise NotImplementedError


def compute_roc_auc(probs, targets):
    """Compute ROC-AUC score.

    Args:
        probs: Predicted probabilities (N,)
        targets: Ground truth labels (N,)

    Returns:
        ROC-AUC score
    """
    return roc_auc_score(targets, probs)


def compute_sensitivity_specificity(preds, targets):
    """Compute sensitivity and specificity.

    Args:
        preds: Binary predictions (N,)
        targets: Ground truth labels (N,)

    Returns:
        tuple: (sensitivity, specificity)
    """
    # Sensitivity = TP / (TP + FN)
    # Specificity = TN / (TN + FP)
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return float(sensitivity), float(specificity)


def compute_balanced_accuracy(preds, targets):
    """Compute balanced accuracy.

    Args:
        preds: Binary predictions (N,)
        targets: Ground truth labels (N,)

    Returns:
        Balanced accuracy
    """
    return balanced_accuracy_score(targets, preds)


def expected_calibration_error(probs, targets, n_bins=15):
    """Compute Expected Calibration Error (ECE).

    Args:
        probs: Predicted probabilities (N,)
        targets: Ground truth labels (N,)
        n_bins: Number of bins for calibration

    Returns:
        ECE score
    """
    # Bin predictions by confidence
    # Compute |accuracy - confidence| per bin
    # Weight by bin size
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = targets[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)
