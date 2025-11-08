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
    # TODO: Implement
    # 1. Instantiate backbone (UNetRes34 or DeepLabV3+)
    # 2. Instantiate loss (DiceBCE, Tversky, etc.)
    # 3. Wrap in LightningSegModel
    raise NotImplementedError


def build_classification_model(config: dict):
    """Build classification model from config.

    Args:
        config: Configuration dictionary

    Returns:
        Lightning model
    """
    # TODO: Implement
    # 1. Instantiate backbone (ResNet18 or EfficientNet)
    # 2. Instantiate loss (Focal, CE, etc.)
    # 3. Wrap in LightningClsModel
    raise NotImplementedError


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

    # DataModule
    if config["task"] == "segmentation":
        datamodule = BusUcSegDataModule(
            data_dir=config["data_dir"],
            img_size=config["img_size"],
            batch_size=config["batch_size"],
            augment_level=config["augment"],
            fold=fold,
            n_folds=config["kfolds"],
            seed=config["seed"],
        )
        model = build_segmentation_model(config)
    elif config["task"] == "classification":
        datamodule = BusUcClsDataModule(
            data_dir=config["data_dir"],
            img_size=config["img_size"],
            batch_size=config["batch_size"],
            augment_level=config["augment"],
            fold=fold,
            n_folds=config["kfolds"],
            seed=config["seed"],
        )
        model = build_classification_model(config)
    else:
        raise ValueError(f"Unknown task: {config['task']}")

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

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Test
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # Save metrics
    # TODO: Extract and save metrics to JSON

    console.print(f"[bold green]✓ Fold {fold} complete[/bold green]")


def aggregate_cv_results(output_dir: Path, n_folds: int):
    """Aggregate cross-validation results.

    Args:
        output_dir: Directory with fold results
        n_folds: Number of folds
    """
    # TODO: Implement
    # 1. Load metrics from each fold
    # 2. Compute mean ± std
    # 3. Save to aggregated JSON
    # 4. Print summary table
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
