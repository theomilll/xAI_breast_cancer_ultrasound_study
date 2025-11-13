"""ResNet-18 classification model for breast ultrasound."""

import torch
import torch.nn as nn
import lightning as L
import torchvision.models as models
from torchmetrics import Accuracy, AUROC, Precision, Recall, ConfusionMatrix


class ResNet18Classifier(nn.Module):
    """ResNet-18 for binary classification (benign vs malignant).

    Uses ImageNet pretrained weights.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            num_classes: Number of output classes (2 for binary)
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze early layers for transfer learning
        """
        super().__init__()

        # Load pretrained ResNet-18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # Optionally freeze early layers
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if "layer4" not in name and "fc" not in name:
                    param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Logits tensor (B, num_classes)
        """
        return self.backbone(x)

    def unfreeze_all(self):
        """Unfreeze all parameters for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True


class LightningClsModel(L.LightningModule):
    """Lightning wrapper for classification models.

    Handles training/validation/test steps, metrics, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        scheduler: str = "one_cycle",
        epochs: int = 80,
        steps_per_epoch: int = None,
        unfreeze_epoch: int = None,
    ):
        """
        Args:
            model: Classification model
            loss_fn: Loss function (CrossEntropy, Focal, etc.)
            learning_rate: Initial learning rate
            weight_decay: AdamW weight decay
            scheduler: LR scheduler ('one_cycle', 'cosine', or None)
            epochs: Total epochs for scheduler
            steps_per_epoch: Steps per epoch for OneCycleLR
            unfreeze_epoch: Epoch to unfreeze backbone (if frozen)
        """
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.unfreeze_epoch = unfreeze_epoch

        # Metrics
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")

        # Save hyperparameters
        self.save_hyperparameters(ignore=["model", "loss_fn"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        images = batch["image"]
        labels = batch["label"]

        # Forward pass
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        # Compute metrics
        probs = torch.softmax(logits, dim=1)[:, 1]  # Prob of class 1 (malignant)
        preds = torch.argmax(logits, dim=1)

        self.train_acc(preds, labels)
        self.train_auroc(probs, labels)

        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images = batch["image"]
        labels = batch["label"]

        # Forward pass
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        # Compute metrics
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        self.val_acc(preds, labels)
        self.val_auroc(probs, labels)

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_roc_auc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        images = batch["image"]
        labels = batch["label"]

        # Forward pass
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        # Compute metrics
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        self.test_acc(preds, labels)
        self.test_auroc(probs, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_roc_auc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        # Unfreeze backbone at specified epoch
        if self.unfreeze_epoch and self.current_epoch == self.unfreeze_epoch:
            self.model.unfreeze_all()
            print(f"Unfreezing backbone at epoch {self.current_epoch}")

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if self.scheduler == "one_cycle":
            if self.steps_per_epoch is None:
                raise ValueError("steps_per_epoch required for OneCycleLR")
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.epochs * self.steps_per_epoch,
                pct_start=0.3,
                anneal_strategy="cos",
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        elif self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epochs,
                eta_min=1e-6,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer
