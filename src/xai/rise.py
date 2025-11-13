"""RISE: Randomized Input Sampling for Explanation.

Black-box saliency method that doesn't require gradients.
Works for both segmentation and classification models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from tqdm import tqdm


class RISE:
    """RISE: Black-box saliency method using random masks.

    Generates saliency by computing expected output over random occlusions.
    More robust than gradient methods but computationally expensive.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_size: tuple[int, int] = (256, 256),
        n_masks: int = 800,
        mask_prob: float = 0.5,
        cell_size: int = 8,
        seed: int = 42,
    ):
        """
        Args:
            model: Segmentation model
            input_size: Input image size (H, W)
            n_masks: Number of random masks to sample (400-1000)
            mask_prob: Probability of keeping each cell
            cell_size: Size of mask cells before upsampling (7-9 px typical)
            seed: Random seed for reproducibility
        """
        self.model = model
        self.input_size = input_size
        self.n_masks = n_masks
        self.mask_prob = mask_prob
        self.cell_size = cell_size
        self.seed = seed
        self.masks = None  # Cache masks

    def generate_masks(self, device: torch.device) -> torch.Tensor:
        """Generate random binary masks.

        Returns:
            Tensor of masks (n_masks, 1, H, W) in [0, 1]
        """
        if self.masks is not None and self.masks.device == device:
            return self.masks

        # Set seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        H, W = self.input_size
        # Low-res grid size
        h_grid = H // self.cell_size
        w_grid = W // self.cell_size

        # Generate random binary masks at low resolution
        masks_low = torch.rand(self.n_masks, 1, h_grid, w_grid, device=device)
        masks_low = (masks_low < self.mask_prob).float()

        # Upsample to input size with bilinear interpolation
        masks = F.interpolate(
            masks_low,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        # Cache masks
        self.masks = masks
        return masks

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate RISE saliency map.

        Args:
            input_tensor: Input image (1, C, H, W)
            target_mask: Optional GT mask for segmentation (mutually exclusive with target_class)
            target_class: Optional target class for classification (mutually exclusive with target_mask)
            batch_size: Batch size for masked forward passes
            normalize: Normalize output to [0, 1]
            show_progress: Show progress bar

        Returns:
            Saliency map (H, W) normalized to [0, 1] if specified
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Generate masks
        masks = self.generate_masks(device)  # (n_masks, 1, H, W)
        n_masks = masks.shape[0]

        # Storage for scores
        scores = torch.zeros(n_masks, device=device)

        # Batch process masked inputs
        iterator = range(0, n_masks, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="RISE", total=(n_masks + batch_size - 1) // batch_size)

        with torch.no_grad():
            for i in iterator:
                batch_masks = masks[i : i + batch_size]  # (B, 1, H, W)
                batch_size_actual = batch_masks.shape[0]

                # Apply masks to input
                masked_inputs = input_tensor * batch_masks  # (B, C, H, W)

                # Forward pass
                logits = self.model(masked_inputs)

                # Compute scores
                if target_class is not None:
                    # Classification: score w.r.t. target class probability
                    # logits shape: (B, num_classes)
                    probs = F.softmax(logits, dim=1)
                    batch_scores = probs[:, target_class]
                elif target_mask is not None:
                    # Segmentation: score w.r.t. GT mask
                    # logits shape: (B, 1, H, W)
                    mask = target_mask.to(device)
                    batch_scores = (logits * mask).sum(dim=(1, 2, 3))
                else:
                    # Segmentation: score as sum of predicted foreground
                    probs = torch.sigmoid(logits)
                    batch_scores = probs.sum(dim=(1, 2, 3))

                scores[i : i + batch_size_actual] = batch_scores

        # Compute weighted average saliency: S = Σ (score_i * mask_i) / n_masks
        # This gives expected saliency when that region is revealed
        scores = scores.view(-1, 1, 1, 1)  # (n_masks, 1, 1, 1)
        saliency = (scores * masks).sum(dim=0).squeeze()  # (H, W)

        # Normalize by number of masks
        saliency = saliency / n_masks

        # Convert to numpy
        saliency = saliency.cpu().numpy()

        # Normalize to [0, 1]
        if normalize and saliency.max() > saliency.min():
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

        return saliency


def compute_rise(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_mask: Optional[torch.Tensor] = None,
    target_class: Optional[int] = None,
    n_masks: int = 800,
    cell_size: int = 8,
    batch_size: int = 32,
    seed: int = 42,
) -> np.ndarray:
    """Convenience function to compute RISE saliency in one call.

    Args:
        model: Segmentation or classification model
        input_tensor: Input image (1, C, H, W)
        target_mask: Optional GT mask (segmentation only)
        target_class: Optional target class (classification only)
        n_masks: Number of random masks (400-1000)
        cell_size: Mask grid cell size (7-9 typical)
        batch_size: Batch size for forward passes
        seed: Random seed

    Returns:
        Saliency map (H, W) normalized to [0, 1]
    """
    H, W = input_tensor.shape[2:]
    rise = RISE(
        model=model,
        input_size=(H, W),
        n_masks=n_masks,
        cell_size=cell_size,
        seed=seed,
    )
    return rise.generate(
        input_tensor=input_tensor,
        target_mask=target_mask,
        target_class=target_class,
        batch_size=batch_size,
        normalize=True,
    )
