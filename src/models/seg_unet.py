"""U-Net segmentation model using segmentation_models_pytorch."""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetRes34(nn.Module):
    """U-Net with ResNet-34 encoder for breast ultrasound segmentation.

    Uses ImageNet pretrained weights.
    """

    def __init__(
        self,
        num_classes: int = 1,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
    ):
        """
        Args:
            num_classes: Number of output channels (1 for binary segmentation)
            encoder_name: Encoder backbone name
            encoder_weights: Pretrained weights ('imagenet' or None)
            in_channels: Number of input channels (3 for RGB, 1 for grayscale)
        """
        super().__init__()
        # TODO: Initialize smp.Unet with specified parameters
        # self.model = smp.Unet(...)
        raise NotImplementedError

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Logits tensor (B, num_classes, H, W)
        """
        # TODO: Forward pass through model
        raise NotImplementedError


class LightningSegModel(nn.Module):
    """Lightning wrapper for segmentation models.

    Handles training/validation steps, metrics, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = "cosine",
        epochs: int = 100,
    ):
        """
        Args:
            model: Segmentation model
            loss_fn: Loss function
            learning_rate: Initial learning rate
            weight_decay: AdamW weight decay
            scheduler: LR scheduler ('cosine', 'one_cycle', or None)
            epochs: Total epochs for scheduler
        """
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.epochs = epochs

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        # TODO: Implement
        # - Forward pass
        # - Compute loss
        # - Compute metrics (Dice, IoU)
        # - Log metrics
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # TODO: Implement similar to training_step
        raise NotImplementedError

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # TODO: Implement
        # - AdamW optimizer
        # - CosineAnnealingLR or OneCycleLR scheduler
        raise NotImplementedError
