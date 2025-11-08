"""Albumentations transforms for ultrasound images."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_train_transforms(img_size: int = 256, augment_level: str = "medium"):
    """Get training transforms with augmentation.

    Args:
        img_size: Target image size (square)
        augment_level: 'light', 'medium', or 'heavy'

    Returns:
        Albumentations Compose object
    """
    # Define augmentation probabilities based on level
    if augment_level == "light":
        flip_p, rotate_p, elastic_p, brightness_p, speckle_p = 0.3, 0.3, 0.1, 0.2, 0.2
    elif augment_level == "medium":
        flip_p, rotate_p, elastic_p, brightness_p, speckle_p = 0.5, 0.5, 0.3, 0.3, 0.3
    elif augment_level == "heavy":
        flip_p, rotate_p, elastic_p, brightness_p, speckle_p = 0.7, 0.7, 0.5, 0.5, 0.5
    else:
        raise ValueError(f"Unknown augment_level: {augment_level}")

    transforms = [
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=flip_p),
        A.Rotate(limit=15, p=rotate_p, border_mode=0),
        A.ElasticTransform(alpha=10, sigma=4, alpha_affine=0, p=elastic_p),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=brightness_p),
        A.Lambda(image=lambda img, **kwargs: SpeckleNoise(intensity=0.1, p=speckle_p)(img)),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def get_val_transforms(img_size: int = 256):
    """Get validation/test transforms (no augmentation).

    Args:
        img_size: Target image size (square)

    Returns:
        Albumentations Compose object
    """
    transforms = [
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


class SpeckleNoise:
    """Add speckle noise to ultrasound images (multiplicative Rayleigh noise)."""

    def __init__(self, intensity: float = 0.1, p: float = 0.5):
        """
        Args:
            intensity: Noise intensity (std of Rayleigh distribution)
            p: Probability of applying transform
        """
        self.intensity = intensity
        self.p = p

    def __call__(self, image, **kwargs):
        """Apply speckle noise.

        Args:
            image: Input image (H, W, C) or (H, W)

        Returns:
            Noisy image
        """
        if np.random.rand() > self.p:
            return image

        # Generate Rayleigh noise (multiplicative)
        noise = np.random.rayleigh(scale=self.intensity, size=image.shape)

        # Apply multiplicative noise
        noisy_image = image * (1 + noise)

        # Clip to valid range (assume 0-255 for uint8 or 0-1 for float)
        if image.dtype == np.uint8:
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        else:
            noisy_image = np.clip(noisy_image, 0, 1).astype(np.float32)

        return noisy_image
