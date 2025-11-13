"""Integrated Gradients implementation for segmentation and classification models.

Uses Captum library for efficient IG computation with various baselines.
"""

import torch
import numpy as np
from typing import Optional, Literal
from captum.attr import IntegratedGradients, NoiseTunnel


def compute_integrated_gradients(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_mask: Optional[torch.Tensor] = None,
    target_class: Optional[int] = None,
    baseline: Literal["black", "mean", "blur"] = "black",
    n_steps: int = 50,
    use_smoothgrad: bool = False,
    smoothgrad_samples: int = 25,
    normalize: bool = True,
) -> np.ndarray:
    """Compute Integrated Gradients attribution for segmentation or classification model.

    Args:
        model: Segmentation or classification model
        input_tensor: Input image (1, C, H, W)
        target_mask: Optional GT mask for segmentation (mutually exclusive with target_class)
        target_class: Optional target class for classification (mutually exclusive with target_mask)
        baseline: Baseline type ("black", "mean", or "blur")
        n_steps: Number of interpolation steps (higher = more accurate but slower)
        use_smoothgrad: Apply SmoothGrad to reduce noise
        smoothgrad_samples: Number of noise samples for SmoothGrad
        normalize: Normalize output to [0, 1]

    Returns:
        Saliency map (H, W) normalized to [0, 1] if specified
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Prepare baseline
    if baseline == "black":
        baseline_tensor = torch.zeros_like(input_tensor)
    elif baseline == "mean":
        # Mean color baseline (computed per channel)
        baseline_tensor = input_tensor.mean(dim=(2, 3), keepdim=True).expand_as(input_tensor)
    elif baseline == "blur":
        # Gaussian blur baseline (simple approximation)
        baseline_tensor = torch.nn.functional.avg_pool2d(
            input_tensor, kernel_size=15, stride=1, padding=7
        )
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    # Define forward function that returns scalar target
    def forward_func(x):
        logits = model(x)
        if target_class is not None:
            # Classification: return target class logit
            # logits shape: (B, num_classes)
            return logits[:, target_class]
        elif target_mask is not None:
            # Segmentation: compute score w.r.t. GT mask
            # logits shape: (B, 1, H, W)
            mask = target_mask.to(logits.device)
            return (logits * mask).sum(dim=(1, 2, 3))
        else:
            # Segmentation: use predicted foreground
            probs = torch.sigmoid(logits)
            return probs.sum(dim=(1, 2, 3))

    # Initialize IntegratedGradients
    ig = IntegratedGradients(forward_func)

    # Optionally wrap with SmoothGrad
    if use_smoothgrad:
        ig = NoiseTunnel(ig)
        attributions = ig.attribute(
            input_tensor,
            baselines=baseline_tensor,
            n_steps=n_steps,
            nt_samples=smoothgrad_samples,
            nt_type="smoothgrad",
        )
    else:
        attributions = ig.attribute(
            input_tensor,
            baselines=baseline_tensor,
            n_steps=n_steps,
        )

    # Aggregate across channels (L2 norm or absolute sum)
    # Using L2 norm for better localization
    attributions = attributions.squeeze(0)  # (C, H, W)
    saliency = torch.sqrt(torch.sum(attributions ** 2, dim=0))  # (H, W)

    # Convert to numpy
    saliency = saliency.detach().cpu().numpy()

    # Normalize to [0, 1]
    if normalize and saliency.max() > 0:
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    return saliency


class IntegratedGradientsExplainer:
    """Wrapper class for consistent interface with other XAI methods."""

    def __init__(
        self,
        model: torch.nn.Module,
        baseline: Literal["black", "mean", "blur"] = "black",
        n_steps: int = 50,
        use_smoothgrad: bool = False,
    ):
        """
        Args:
            model: Segmentation model
            baseline: Baseline type
            n_steps: Number of interpolation steps
            use_smoothgrad: Apply SmoothGrad
        """
        self.model = model
        self.baseline = baseline
        self.n_steps = n_steps
        self.use_smoothgrad = use_smoothgrad

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Generate IG saliency map.

        Args:
            input_tensor: Input image (1, C, H, W)
            target_mask: Optional GT mask (segmentation only)
            target_class: Optional target class (classification only)
            normalize: Normalize output

        Returns:
            Saliency map (H, W) in [0, 1] if normalized
        """
        return compute_integrated_gradients(
            model=self.model,
            input_tensor=input_tensor,
            target_mask=target_mask,
            target_class=target_class,
            baseline=self.baseline,
            n_steps=self.n_steps,
            use_smoothgrad=self.use_smoothgrad,
            normalize=normalize,
        )


def compare_baselines(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_mask: Optional[torch.Tensor] = None,
    target_class: Optional[int] = None,
    n_steps: int = 50,
) -> dict[str, np.ndarray]:
    """Compare IG attributions with different baselines.

    Useful for sanity checking and understanding baseline sensitivity.

    Args:
        model: Segmentation or classification model
        input_tensor: Input image (1, C, H, W)
        target_mask: Optional GT mask (segmentation only)
        target_class: Optional target class (classification only)
        n_steps: Number of steps

    Returns:
        Dictionary mapping baseline names to saliency maps
    """
    results = {}

    for baseline in ["black", "mean", "blur"]:
        saliency = compute_integrated_gradients(
            model=model,
            input_tensor=input_tensor,
            target_mask=target_mask,
            target_class=target_class,
            baseline=baseline,
            n_steps=n_steps,
            normalize=True,
        )
        results[baseline] = saliency

    return results
