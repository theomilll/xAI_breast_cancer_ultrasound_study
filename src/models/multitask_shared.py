"""Multi-task model with shared encoder for segmentation + classification."""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class MultiTaskModel(nn.Module):
    """Shared encoder with U-Net decoder and classification head.

    Enables joint training of segmentation and classification.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        seg_classes: int = 1,
        cls_classes: int = 1,
    ):
        """
        Args:
            encoder_name: Encoder backbone
            encoder_weights: Pretrained weights
            seg_classes: Number of segmentation classes
            cls_classes: Number of classification classes
        """
        super().__init__()
        # TODO: Implement
        # - Shared encoder (from smp)
        # - U-Net decoder for segmentation
        # - Global pooling + FC head for classification
        raise NotImplementedError

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            tuple: (seg_logits, cls_logits)
        """
        # TODO: Forward through encoder, both heads
        raise NotImplementedError
