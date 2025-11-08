"""RISE: Randomized Input Sampling for Explanation."""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class RISE:
    """RISE: Black-box saliency method using random masks.

    Generates saliency by computing expected output over random occlusions.
    """

    def __init__(
        self,
        model,
        input_size=(256, 256),
        n_masks=800,
        mask_prob=0.5,
        cell_size=8,
    ):
        """
        Args:
            model: Neural network model
            input_size: Input image size (H, W)
            n_masks: Number of random masks to sample
            mask_prob: Probability of keeping each cell in mask
            cell_size: Size of mask cells before upsampling
        """
        self.model = model
        self.input_size = input_size
        self.n_masks = n_masks
        self.mask_prob = mask_prob
        self.cell_size = cell_size

    def generate_masks(self):
        """Generate random binary masks.

        Returns:
            Tensor of masks (n_masks, 1, H, W)
        """
        # TODO: Implement
        # 1. Generate low-res binary masks (H/cell_size, W/cell_size)
        # 2. Upsample to input_size with bilinear interpolation
        # 3. Apply smoothing
        raise NotImplementedError

    def generate(self, input_tensor, target_class=None, batch_size=16):
        """Generate RISE saliency map.

        Args:
            input_tensor: Input image (1, C, H, W)
            target_class: Target class index (None = predicted)
            batch_size: Batch size for masked forward passes

        Returns:
            Saliency map (H, W) normalized to [0, 1]
        """
        # TODO: Implement
        # 1. Generate or load precomputed masks
        # 2. Mask input: masked_inputs = input * masks
        # 3. Batch forward passes through model
        # 4. Compute expected saliency: S = Σ (score_i * mask_i) / Σ score_i
        # 5. Normalize
        raise NotImplementedError
