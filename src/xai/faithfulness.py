"""Faithfulness metrics for XAI evaluation."""

import torch
import numpy as np
from typing import Callable


def insertion_deletion_curves(
    model,
    input_tensor,
    saliency_map,
    target_class=None,
    n_steps=100,
    mode="insertion",
):
    """Compute insertion or deletion curve.

    Insertion: progressively reveal high-saliency regions (start from blank)
    Deletion: progressively hide high-saliency regions (start from full image)

    Args:
        model: Neural network model
        input_tensor: Input image (1, C, H, W)
        saliency_map: Saliency map (H, W) or (1, 1, H, W)
        target_class: Target class index
        n_steps: Number of steps (pixels to reveal/delete per step)
        mode: 'insertion' or 'deletion'

    Returns:
        Array of scores at each step (n_steps,)
    """
    # TODO: Implement
    # 1. Sort pixels by saliency
    # 2. For each step:
    #    - Reveal/delete top-k pixels
    #    - Forward pass to get score
    # 3. Return score trajectory
    raise NotImplementedError


def insertion_auc(model, input_tensor, saliency_map, target_class=None, n_steps=100):
    """Compute area under insertion curve.

    Higher is better (faster rise to final score).
    """
    # TODO: Implement using insertion_deletion_curves
    raise NotImplementedError


def deletion_auc(model, input_tensor, saliency_map, target_class=None, n_steps=100):
    """Compute area under deletion curve.

    Lower is better (faster drop from initial score).
    """
    # TODO: Implement using insertion_deletion_curves
    raise NotImplementedError


def pointing_game(saliency_map, ground_truth_mask, threshold=0.5):
    """Pointing Game: Does peak saliency fall inside ground truth mask?

    Args:
        saliency_map: Saliency map (H, W)
        ground_truth_mask: Binary mask (H, W)
        threshold: Mask threshold for binarization

    Returns:
        bool: True if max saliency is inside mask
    """
    # TODO: Implement
    # 1. Find location of max saliency
    # 2. Check if that location is inside GT mask
    raise NotImplementedError


def saliency_iou(saliency_map, ground_truth_mask, top_k_percent=20):
    """Compute IoU between top-k% saliency and ground truth mask.

    Args:
        saliency_map: Saliency map (H, W)
        ground_truth_mask: Binary mask (H, W)
        top_k_percent: Percentage of top saliency to consider

    Returns:
        IoU score
    """
    # TODO: Implement
    # 1. Threshold saliency map to keep top k% pixels
    # 2. Binarize mask
    # 3. Compute IoU
    raise NotImplementedError


def randomization_test(model, input_tensor, saliency_method: Callable, layers_to_randomize):
    """Sanity check: randomizing weights should degrade saliency.

    Args:
        model: Neural network model
        input_tensor: Input image
        saliency_method: Function that generates saliency map
        layers_to_randomize: List of layer names to progressively randomize

    Returns:
        List of saliency maps (one per randomization level)
    """
    # TODO: Implement
    # 1. Generate baseline saliency
    # 2. For each layer:
    #    - Randomize weights
    #    - Generate saliency
    # 3. Return trajectory of saliency maps
    raise NotImplementedError
