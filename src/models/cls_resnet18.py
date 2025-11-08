"""ResNet-18 classification model for benign vs malignant."""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Classifier(nn.Module):
    """ResNet-18 with custom head for binary classification."""

    def __init__(
        self,
        num_classes: int = 1,  # 1 for binary with BCE, 2 for CE
        pretrained: bool = True,
        freeze_epochs: int = 0,
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            freeze_epochs: Number of epochs to freeze early layers
        """
        super().__init__()
        # TODO: Initialize ResNet-18
        # - Load pretrained weights
        # - Replace final FC layer
        # - Optionally freeze early blocks
        raise NotImplementedError

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Logits tensor (B, num_classes)
        """
        # TODO: Forward pass
        raise NotImplementedError

    def unfreeze(self):
        """Unfreeze all layers for fine-tuning."""
        # TODO: Set requires_grad=True for all parameters
        raise NotImplementedError


class LightningClsModel(nn.Module):
    """Lightning wrapper for classification models.

    Handles training/validation steps, metrics, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        scheduler: str = "one_cycle",
        epochs: int = 80,
    ):
        """
        Args:
            model: Classification model
            loss_fn: Loss function (Focal, BCE, or CE)
            learning_rate: Initial learning rate
            weight_decay: AdamW weight decay
            scheduler: LR scheduler
            epochs: Total epochs
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
        # - Compute metrics (accuracy, AUC)
        # - Log metrics
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # TODO: Implement
        # - Compute ROC-AUC, sensitivity, specificity, balanced acc
        raise NotImplementedError

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # TODO: Implement AdamW + OneCycleLR
        raise NotImplementedError
