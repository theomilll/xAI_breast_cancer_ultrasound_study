"""U-Net segmentation model using segmentation_models_pytorch."""

import torch
import torch.nn as nn
import lightning as L
import segmentation_models_pytorch as smp

from src.metrics import dice_score, iou_score


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
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,  # Return logits
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Logits tensor (B, num_classes, H, W)
        """
        return self.model(x)


class LightningSegModel(L.LightningModule):
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
        images = batch["image"]
        masks = batch["binary_mask"]

        # Forward pass
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        # Compute metrics
        preds = (torch.sigmoid(logits) > 0.5).float()
        dice = dice_score(preds, masks)
        iou = iou_score(preds, masks)

        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_dice", dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_iou", iou, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images = batch["image"]
        masks = batch["binary_mask"]

        # Forward pass
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        # Compute metrics
        preds = (torch.sigmoid(logits) > 0.5).float()
        dice = dice_score(preds, masks)
        iou = iou_score(preds, masks)

        # Log metrics (val_dice is monitored by checkpoint callback)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_dice", dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_iou", iou, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        images = batch["image"]
        masks = batch["binary_mask"]

        # Forward pass
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        # Compute metrics
        preds = (torch.sigmoid(logits) > 0.5).float()
        dice = dice_score(preds, masks)
        iou = iou_score(preds, masks)

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_dice", dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_iou", iou, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epochs,
                eta_min=1e-6,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.scheduler == "one_cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.epochs,
                pct_start=0.3,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer
