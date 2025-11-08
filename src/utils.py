"""Utility functions for data loading, seeding, and visualization."""

import random
import numpy as np
import torch
from pathlib import Path


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_binary_mask(mask):
    """Convert grayscale mask to binary (0 or 1).

    Args:
        mask: Grayscale mask array (H, W) or (H, W, C)

    Returns:
        Binary mask (H, W) with values 0 or 1
    """
    gray = mask[..., 0] if mask.ndim == 3 else mask
    return (gray > 0).astype(np.uint8)


def get_data_paths(data_dir: Path, task: str = "segmentation"):
    """Get paths to images and masks/labels based on task.

    Args:
        data_dir: Root data directory
        task: Either 'segmentation' or 'classification'

    Returns:
        Dictionary with benign and malignant paths
    """
    data_dir = Path(data_dir)

    if task == "segmentation":
        # Navigate nested structure: BUS_UC/BUS_UC/BUS_UC/
        nested_dir = data_dir
        while (nested_dir / "BUS_UC").exists():
            nested_dir = nested_dir / "BUS_UC"

        # Get paths for both classes
        benign_img_dir = nested_dir / "Benign" / "images"
        benign_mask_dir = nested_dir / "Benign" / "masks"
        malignant_img_dir = nested_dir / "Malignant" / "images"
        malignant_mask_dir = nested_dir / "Malignant" / "masks"

        # Get sorted file lists
        benign_images = sorted(list(benign_img_dir.glob("*.png")))
        benign_masks = sorted(list(benign_mask_dir.glob("*.png")))
        malignant_images = sorted(list(malignant_img_dir.glob("*.png")))
        malignant_masks = sorted(list(malignant_mask_dir.glob("*.png")))

        return {
            "benign": {"images": benign_images, "masks": benign_masks},
            "malignant": {"images": malignant_images, "masks": malignant_masks}
        }

    elif task == "classification":
        # Navigate nested structure: BUS_UC_classification/BUS_UC_classification/
        nested_dir = data_dir
        while (nested_dir / "BUS_UC_classification").exists():
            nested_dir = nested_dir / "BUS_UC_classification"

        # Get paths for both classes
        benign_dir = nested_dir / "Benign"
        malignant_dir = nested_dir / "Malignant"

        benign_images = sorted(list(benign_dir.glob("*.png")))
        malignant_images = sorted(list(malignant_dir.glob("*.png")))

        return {
            "benign": {"images": benign_images},
            "malignant": {"images": malignant_images}
        }

    else:
        raise ValueError(f"Unknown task: {task}. Must be 'segmentation' or 'classification'")


def compute_lesion_area(mask):
    """Compute lesion area as percentage of total image.

    Args:
        mask: Binary mask array

    Returns:
        Lesion area percentage (0-100)
    """
    if mask.size == 0:
        return 0.0

    lesion_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    area_percentage = (lesion_pixels / total_pixels) * 100.0

    return area_percentage
