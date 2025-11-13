"""Main training script with Lightning Trainer and CLI."""

import argparse
from pathlib import Path
import yaml
import json

import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import CSVLogger
from rich.console import Console

from src.utils import seed_everything
from src.datamodules.bus_uc import BusUcSegDataModule, BusUcClsDataModule
from src.models.seg_unet import UNetRes34, LightningSegModel
from src.models.cls_resnet18 import ResNet18Classifier, LightningClsModel
from src.losses import DiceBCELoss, FocalLoss
from src.metrics import dice_score, iou_score


console = Console()


def load_config(config_path: str) -> dict:
    """Load YAML config file.

    Args:
        config_path: Path to YAML config

    Returns:
        Config dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_segmentation_model(config: dict):
    """Build segmentation model from config.

    Args:
        config: Configuration dictionary

    Returns:
        Lightning model
    """
    # Instantiate backbone
    if config["model"] == "unet_res34":
        backbone = UNetRes34(
            num_classes=1,
            encoder_name="resnet34",
            encoder_weights="imagenet",
        )
    else:
        raise ValueError(f"Unknown model: {config['model']}")

    # Instantiate loss
    loss_config = config["loss"]
    if loss_config["name"] == "dice_bce":
        loss_fn = DiceBCELoss(
            dice_weight=loss_config.get("dice_weight", 0.5),
            bce_weight=loss_config.get("bce_weight", 0.5),
        )
    else:
        raise ValueError(f"Unknown loss: {loss_config['name']}")

    # Wrap in Lightning model
    model = LightningSegModel(
        model=backbone,
        loss_fn=loss_fn,
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        scheduler=config["scheduler"],
        epochs=config["epochs"],
    )

    return model


def build_classification_model(config: dict, steps_per_epoch: int):
    """Build classification model from config.

    Args:
        config: Configuration dictionary
        steps_per_epoch: Steps per epoch for OneCycleLR

    Returns:
        Lightning model
    """
    # Instantiate backbone
    if config["model"] == "resnet18":
        backbone = ResNet18Classifier(
            num_classes=config["num_classes"],
            pretrained=config.get("pretrained", True),
            freeze_backbone=config.get("freeze_backbone", False),
        )
    else:
        raise ValueError(f"Unknown model: {config['model']}")

    # Instantiate loss
    loss_config = config["loss"]
    if loss_config["name"] == "focal":
        loss_fn = FocalLoss(
            gamma=loss_config.get("gamma", 2.0),
            alpha=loss_config.get("alpha", None),
        )
    elif loss_config["name"] == "cross_entropy":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss: {loss_config['name']}")

    # Wrap in Lightning model
    model = LightningClsModel(
        model=backbone,
        loss_fn=loss_fn,
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        scheduler=config["scheduler"],
        epochs=config["epochs"],
        steps_per_epoch=steps_per_epoch,
        unfreeze_epoch=config.get("unfreeze_epoch", None),
    )

    return model


def train_single_fold(config: dict, fold: int, output_dir: Path):
    """Train a single fold.

    Args:
        config: Configuration dictionary
        fold: Fold index
        output_dir: Output directory for checkpoints/logs
    """
    console.print(f"[bold blue]Training fold {fold}/{config['kfolds']}[/bold blue]")

    # Seed
    seed_everything(config["seed"] + fold)

    # DataModule - segmentation or classification
    task = config.get("task", "segmentation")
    if task == "segmentation":
        datamodule = BusUcSegDataModule(
            data_dir=config["data_dir"],
            img_size=config["img_size"],
            batch_size=config["batch_size"],
            augment_level=config["augment"],
            fold=fold,
            n_folds=config["kfolds"],
            seed=config["seed"],
        )
    elif task == "classification":
        datamodule = BusUcClsDataModule(
            data_dir=config["data_dir"],
            img_size=config["img_size"],
            batch_size=config["batch_size"],
            augment_level=config["augment"],
            fold=fold,
            n_folds=config["kfolds"],
            use_weighted_sampler=config.get("use_weighted_sampler", True),
            seed=config["seed"],
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    # Setup datamodule to compute dataset size
    datamodule.setup()

    # Build model
    if task == "segmentation":
        model = build_segmentation_model(config)
    elif task == "classification":
        steps_per_epoch = len(datamodule.train_dataloader())
        model = build_classification_model(config, steps_per_epoch)
    else:
        raise ValueError(f"Unknown task: {task}")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / f"fold_{fold}",
        filename="best",
        monitor=f"val_{config['save_best_by']}",
        mode="max" if "dice" in config["save_best_by"] or "auc" in config["save_best_by"] else "min",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor=f"val_{config['save_best_by']}",
        patience=config["early_stop_patience"],
        mode="max" if "dice" in config["save_best_by"] or "auc" in config["save_best_by"] else "min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Logger
    logger = CSVLogger(save_dir=output_dir, name=f"fold_{fold}")

    # Trainer
    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        devices=1,
        precision="16-mixed" if config["mixed_precision"] else 32,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        gradient_clip_val=1.0,
        deterministic=True,
    )

    # Check for existing checkpoint to resume
    checkpoint_dir = output_dir / f"fold_{fold}"
    checkpoint_files = list(checkpoint_dir.glob("best*.ckpt")) if checkpoint_dir.exists() else []
    ckpt_path = None
    if checkpoint_files:
        # Use most recent checkpoint
        ckpt_path = str(max(checkpoint_files, key=lambda p: p.stat().st_mtime))
        console.print(f"[yellow]Resuming from checkpoint: {ckpt_path}[/yellow]")

    # Train
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Test
    test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # Save metrics
    metrics = test_results[0] if test_results else {}
    metrics_path = output_dir / f"fold_{fold}" / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    console.print(f"[bold green]✓ Fold {fold} complete[/bold green]")
    console.print(f"  Metrics saved to {metrics_path}")


def aggregate_cv_results(output_dir: Path, n_folds: int):
    """Aggregate cross-validation results.

    Args:
        output_dir: Directory with fold results
        n_folds: Number of folds
    """
    import numpy as np

    # Load metrics from each fold
    all_metrics = []
    for fold in range(n_folds):
        metrics_path = output_dir / f"fold_{fold}" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                all_metrics.append(json.load(f))
        else:
            console.print(f"[yellow]Warning: {metrics_path} not found[/yellow]")

    if not all_metrics:
        console.print("[red]No metrics found to aggregate[/red]")
        return

    # Compute mean ± std
    aggregated = {}
    metric_keys = all_metrics[0].keys()

    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_values"] = values

    # Save aggregated results
    agg_path = output_dir / "cv_results.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    # Print summary table
    console.print("\n[bold cyan]Cross-Validation Results Summary:[/bold cyan]")
    console.print("=" * 60)
    for key in sorted(metric_keys):
        if key in [m for m in metric_keys]:
            mean = aggregated.get(f"{key}_mean", 0)
            std = aggregated.get(f"{key}_std", 0)
            console.print(f"  {key:30s}: {mean:.4f} ± {std:.4f}")
    console.print("=" * 60)
    console.print(f"Results saved to {agg_path}")
    console.print("[bold green]Cross-validation complete![/bold green]")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train BUS_UC models")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config["seed"] = args.seed
    output_dir = Path(args.output) / Path(args.config).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    console.print(f"[bold]Task:[/bold] {config['task']}")
    console.print(f"[bold]Model:[/bold] {config['model']}")
    console.print(f"[bold]Output:[/bold] {output_dir}")

    # Train all folds
    for fold in range(config["kfolds"]):
        train_single_fold(config, fold, output_dir)

    # Aggregate
    aggregate_cv_results(output_dir, config["kfolds"])


if __name__ == "__main__":
    main()
