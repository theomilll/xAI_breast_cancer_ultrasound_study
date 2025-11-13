"""Faithfulness metrics for XAI evaluation.

Measures how well saliency maps reflect true model behavior.
Works for both segmentation and classification models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Literal, Callable
from tqdm import tqdm
import copy
from scipy.stats import spearmanr


def _to_numpy(array):
    """Convert tensors/lists to numpy arrays for downstream metrics."""
    if array is None:
        return None
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    if isinstance(array, np.ndarray):
        return array
    return np.asarray(array)


def insertion_deletion_curves(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    saliency_map: np.ndarray,
    target_mask: Optional[torch.Tensor] = None,
    target_class: Optional[int] = None,
    n_steps: int = 20,
    mode: Literal["insertion", "deletion"] = "insertion",
    baseline: Literal["blur", "mean", "black"] = "blur",
    show_progress: bool = False,
) -> np.ndarray:
    """Compute insertion or deletion curve.

    Insertion: progressively reveal high-saliency regions (start from baseline)
    Deletion: progressively hide high-saliency regions (start from full image)

    Args:
        model: Segmentation or classification model
        input_tensor: Input image (1, C, H, W)
        saliency_map: Saliency map (H, W) normalized to [0, 1]
        target_mask: Optional GT mask for segmentation (mutually exclusive with target_class)
        target_class: Optional target class for classification (mutually exclusive with target_mask)
        n_steps: Number of steps (pixels to reveal/delete per step)
        mode: 'insertion' or 'deletion'
        baseline: Baseline image ('blur', 'mean', or 'black')
        show_progress: Show progress bar

    Returns:
        Array of scores at each step (n_steps,)
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    H, W = input_tensor.shape[2:]
    total_pixels = H * W

    # Prepare baseline
    if baseline == "blur":
        baseline_img = F.avg_pool2d(input_tensor, kernel_size=21, stride=1, padding=10)
    elif baseline == "mean":
        baseline_img = input_tensor.mean(dim=(2, 3), keepdim=True).expand_as(input_tensor)
    elif baseline == "black":
        baseline_img = torch.zeros_like(input_tensor)
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    # Sort pixels by saliency (descending)
    saliency_flat = saliency_map.flatten()
    indices = np.argsort(saliency_flat)[::-1]  # High to low

    # Compute step size
    step_size = max(1, total_pixels // n_steps)

    scores = []

    iterator = range(0, total_pixels, step_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"{mode.capitalize()}", total=n_steps)

    with torch.no_grad():
        for i in iterator:
            # Create mask of revealed pixels
            mask_flat = np.zeros(total_pixels, dtype=np.float32)
            mask_flat[indices[: i + step_size]] = 1.0
            mask = torch.from_numpy(mask_flat.reshape(1, 1, H, W)).to(device)

            if mode == "insertion":
                # Start from baseline, progressively reveal important pixels
                masked_input = baseline_img * (1 - mask) + input_tensor * mask
            else:  # deletion
                # Start from full image, progressively hide important pixels
                masked_input = input_tensor * (1 - mask) + baseline_img * mask

            # Forward pass
            logits = model(masked_input)

            # Compute score
            if target_class is not None:
                # Classification: score w.r.t. target class probability
                probs = F.softmax(logits, dim=1)
                score = probs[0, target_class].item()
            elif target_mask is not None:
                # Segmentation: score w.r.t. GT mask
                gt_mask = target_mask.to(device)
                score = (torch.sigmoid(logits) * gt_mask).sum().item()
            else:
                # Segmentation: sum of predicted foreground
                score = torch.sigmoid(logits).sum().item()

            scores.append(score)

    return np.array(scores)


def insertion_auc(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    saliency_map: np.ndarray,
    target_mask: Optional[torch.Tensor] = None,
    target_class: Optional[int] = None,
    n_steps: int = 20,
) -> float:
    """Compute area under insertion curve.

    Higher is better (faster rise to final score = better saliency).
    Normalized by AUC of oracle (perfect saliency).

    Args:
        model: Segmentation or classification model
        input_tensor: Input image
        saliency_map: Saliency map
        target_mask: Optional GT mask (segmentation only)
        target_class: Optional target class (classification only)
        n_steps: Number of steps

    Returns:
        Normalized AUC in [0, 1] (higher is better)
    """
    curve = insertion_deletion_curves(
        model, input_tensor, saliency_map, target_mask, target_class, n_steps, mode="insertion"
    )

    # Compute AUC using trapezoidal rule
    auc = np.trapz(curve) / len(curve)

    # Normalize by max possible AUC (oracle: final score achieved immediately)
    max_score = curve[-1] if len(curve) > 0 and curve[-1] > 0 else 1.0
    normalized_auc = auc / (max_score * n_steps) if max_score > 0 else 0.0

    return float(normalized_auc)


def deletion_auc(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    saliency_map: np.ndarray,
    target_mask: Optional[torch.Tensor] = None,
    target_class: Optional[int] = None,
    n_steps: int = 20,
) -> float:
    """Compute area under deletion curve.

    Lower is better (faster drop from initial score = better saliency).
    Returns 1 - normalized_auc so higher is better for consistency.

    Args:
        model: Segmentation or classification model
        input_tensor: Input image
        saliency_map: Saliency map
        target_mask: Optional GT mask (segmentation only)
        target_class: Optional target class (classification only)
        n_steps: Number of steps

    Returns:
        1 - normalized AUC (higher is better for consistency)
    """
    curve = insertion_deletion_curves(
        model, input_tensor, saliency_map, target_mask, target_class, n_steps, mode="deletion"
    )

    # Compute AUC
    auc = np.trapz(curve) / len(curve)

    # Normalize by initial score
    initial_score = curve[0] if len(curve) > 0 and curve[0] > 0 else 1.0
    normalized_auc = auc / (initial_score * n_steps) if initial_score > 0 else 0.0

    # Return complement so higher is better
    return float(1.0 - normalized_auc)


def pointing_game(
    saliency_map: np.ndarray,
    ground_truth_mask: np.ndarray,
    threshold: float = 0.5,
) -> bool:
    """Pointing Game: Does peak saliency fall inside ground truth mask?

    Args:
        saliency_map: Saliency map (H, W) in [0, 1]
        ground_truth_mask: Binary mask (H, W) in {0, 1}
        threshold: Mask threshold for binarization

    Returns:
        True if max saliency point is inside mask, False otherwise
    """
    # Find location of max saliency
    max_idx = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)

    # Binarize GT mask
    binary_mask = (ground_truth_mask > threshold).astype(np.uint8)

    # Check if max point is inside mask
    return bool(binary_mask[max_idx] > 0)


def saliency_iou(
    saliency_map: np.ndarray,
    ground_truth_mask: np.ndarray,
    top_k_percent: float = 20.0,
    threshold: float = 0.5,
) -> float:
    """Compute IoU between top-k% saliency and ground truth mask.

    Measures spatial overlap between saliency and true lesion region.

    Args:
        saliency_map: Saliency map (H, W) in [0, 1]
        ground_truth_mask: Binary mask (H, W) in {0, 1}
        top_k_percent: Percentage of top saliency to consider (10-30% typical)
        threshold: GT mask threshold

    Returns:
        IoU score in [0, 1] (higher is better)
    """
    # Binarize GT mask
    gt_binary = (ground_truth_mask > threshold).astype(np.uint8)

    # Threshold saliency to keep top k%
    k = int(saliency_map.size * top_k_percent / 100.0)
    if k == 0:
        return 0.0

    threshold_val = np.partition(saliency_map.flatten(), -k)[-k]
    sal_binary = (saliency_map >= threshold_val).astype(np.uint8)

    # Compute IoU
    intersection = np.logical_and(sal_binary, gt_binary).sum()
    union = np.logical_or(sal_binary, gt_binary).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def evaluate_faithfulness(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    saliency_map: np.ndarray,
    ground_truth_mask: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
    n_steps: int = 20,
    top_k_percents: list[float] = [10, 20, 30],
) -> dict:
    """Evaluate all faithfulness metrics for a saliency map.

    Args:
        model: Segmentation or classification model
        input_tensor: Input image (1, C, H, W)
        saliency_map: Saliency map (H, W) normalized
        ground_truth_mask: Optional GT mask (H, W) for segmentation
        target_class: Optional target class for classification
        n_steps: Steps for insertion/deletion
        top_k_percents: List of k values for saliency IoU

    Returns:
        Dictionary of metric name -> value
    """
    metrics = {}
    saliency_map_np = _to_numpy(saliency_map)

    # Prepare GT mask tensor if provided
    ground_truth_mask_np = _to_numpy(ground_truth_mask)
    gt_mask_tensor = None
    if ground_truth_mask_np is not None:
        gt_mask_tensor = torch.from_numpy(ground_truth_mask_np).float()
        if gt_mask_tensor.dim() == 2:
            gt_mask_tensor = gt_mask_tensor.unsqueeze(0).unsqueeze(0)
        elif gt_mask_tensor.dim() == 3:
            gt_mask_tensor = gt_mask_tensor.unsqueeze(0)
        elif gt_mask_tensor.dim() != 4:
            raise ValueError(
                f"Ground truth mask must have 2-4 dims, got shape {tuple(gt_mask_tensor.shape)}"
            )

    # Insertion/Deletion AUC
    metrics["insertion_auc"] = insertion_auc(
        model, input_tensor, saliency_map_np, gt_mask_tensor, target_class, n_steps
    )
    metrics["deletion_auc"] = deletion_auc(
        model, input_tensor, saliency_map_np, gt_mask_tensor, target_class, n_steps
    )

    # Metrics requiring GT mask
    if ground_truth_mask_np is not None:
        # Pointing Game
        metrics["pointing_game"] = float(pointing_game(saliency_map_np, ground_truth_mask_np))

        # Saliency IoU for multiple thresholds
        for k in top_k_percents:
            metrics[f"saliency_iou_top{int(k)}"] = saliency_iou(
                saliency_map_np, ground_truth_mask_np, top_k_percent=k
            )

    return metrics


def compare_methods(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    saliency_maps: dict[str, np.ndarray],
    ground_truth_mask: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
    n_steps: int = 20,
) -> dict[str, dict]:
    """Compare multiple XAI methods using faithfulness metrics.

    Args:
        model: Segmentation or classification model
        input_tensor: Input image
        saliency_maps: Dict mapping method name -> saliency map
        ground_truth_mask: Optional GT mask (segmentation only)
        target_class: Optional target class (classification only)
        n_steps: Steps for curves

    Returns:
        Dict mapping method name -> metrics dict
    """
    results = {}

    for method_name, saliency_map in saliency_maps.items():
        results[method_name] = evaluate_faithfulness(
            model, input_tensor, saliency_map, ground_truth_mask, target_class, n_steps
        )

    return results


def randomization_test(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    saliency_fn: Callable,
    randomization_levels: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    metric: Literal["spearman", "mse"] = "spearman",
    show_progress: bool = False,
) -> dict:
    """Sanity check: progressively randomize model weights and measure saliency degradation.

    A good saliency method should produce uncorrelated/degraded maps when model weights
    are randomized (Adebayo et al., 2018: "Sanity Checks for Saliency Maps").

    Args:
        model: Model to test
        input_tensor: Input image (1, C, H, W)
        saliency_fn: Function that takes model and input, returns saliency map (H, W)
                     e.g., lambda m, x: compute_gradcam(m, x, target_class=1)
        randomization_levels: Fraction of weights to randomize (0.0 = none, 1.0 = all)
        metric: Correlation metric ('spearman' or 'mse')
        show_progress: Show progress bar

    Returns:
        Dictionary with:
            - 'randomization_levels': list of randomization fractions
            - 'correlations': list of correlation scores vs original saliency
            - 'saliency_maps': dict mapping level -> saliency map
    """
    device = next(model.parameters()).device

    # Generate original saliency map
    model.eval()
    original_saliency = saliency_fn(model, input_tensor)

    results = {
        "randomization_levels": [],
        "correlations": [],
        "saliency_maps": {},
    }

    iterator = randomization_levels
    if show_progress:
        iterator = tqdm(randomization_levels, desc="Randomization test")

    for level in iterator:
        # Create a copy of the model
        model_copy = copy.deepcopy(model)
        model_copy.eval()

        # Randomize weights
        _randomize_model_weights(model_copy, fraction=level)

        # Generate saliency map with randomized model
        randomized_saliency = saliency_fn(model_copy, input_tensor)

        # Compute correlation with original
        if metric == "spearman":
            # Flatten and compute Spearman correlation
            orig_flat = original_saliency.flatten()
            rand_flat = randomized_saliency.flatten()
            correlation, _ = spearmanr(orig_flat, rand_flat)
            correlation = float(correlation) if not np.isnan(correlation) else 0.0
        elif metric == "mse":
            # Compute MSE (lower = more similar)
            mse = np.mean((original_saliency - randomized_saliency) ** 2)
            # Convert to similarity score (higher = more similar)
            correlation = float(1.0 / (1.0 + mse))
        else:
            raise ValueError(f"Unknown metric: {metric}")

        results["randomization_levels"].append(level)
        results["correlations"].append(correlation)
        results["saliency_maps"][f"rand_{int(level*100)}"] = randomized_saliency

        # Clean up
        del model_copy

    return results


def _randomize_model_weights(model: torch.nn.Module, fraction: float = 1.0):
    """Randomize a fraction of model weights in-place (top-down, cascading).

    Args:
        model: Model to randomize
        fraction: Fraction of layers to randomize (0.0 = none, 1.0 = all)
    """
    if fraction == 0.0:
        return

    # Get all parameters
    all_params = list(model.parameters())
    n_total = len(all_params)

    # Determine how many layers to randomize (from top down)
    n_randomize = int(np.ceil(n_total * fraction))

    # Randomize top n_randomize parameters (cascading effect)
    for i, param in enumerate(all_params):
        if i < n_randomize:
            with torch.no_grad():
                # Reinitialize with same shape but random values
                if param.dim() >= 2:
                    # For weights (Conv, Linear, etc.)
                    torch.nn.init.xavier_uniform_(param)
                else:
                    # For biases
                    torch.nn.init.zeros_(param)
