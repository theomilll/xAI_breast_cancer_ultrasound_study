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
    # TODO: Implement
    # Dice = (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    raise NotImplementedError


def iou_score(preds, targets, smooth=1.0):
    """Compute Intersection over Union (Jaccard index).

    Args:
        preds: Binary predictions
        targets: Binary targets
        smooth: Smoothing factor

    Returns:
        IoU score (scalar)
    """
    # TODO: Implement
    # IoU = (|X ∩ Y| + smooth) / (|X ∪ Y| + smooth)
    raise NotImplementedError


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
    # TODO: Implement using confusion matrix
    # Sensitivity = TP / (TP + FN)
    # Specificity = TN / (TN + FP)
    raise NotImplementedError


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
    # TODO: Implement ECE
    # - Bin predictions by confidence
    # - Compute |accuracy - confidence| per bin
    # - Weight by bin size
    raise NotImplementedError
