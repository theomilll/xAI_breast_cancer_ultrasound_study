"""Lightning DataModules and Datasets for BUS_UC."""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import lightning as L
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import json
from typing import Optional, Tuple, List, Union

from src.utils import to_binary_mask, get_data_paths
from src.transforms.ultrasound import get_train_transforms, get_val_transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_repo_path(path: Union[str, Path]) -> Path:
    """Return absolute path, resolving repo-relative strings."""
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    return (PROJECT_ROOT / expanded).resolve()


class BusUcSegDataset(Dataset):
    """Dataset for breast ultrasound segmentation task.

    Returns image, binary_mask, and soft_mask (original grayscale).
    """

    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        transforms=None,
    ):
        """
        Args:
            image_paths: List of paths to ultrasound images
            mask_paths: List of paths to corresponding masks
            transforms: Albumentations transforms
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys: 'image', 'binary_mask', 'soft_mask', 'mask'
        """
        # Load image and mask
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask_gray = np.array(Image.open(self.mask_paths[idx]).convert("L"))

        # Defensive: resize mask if shape mismatch
        if image.shape[:2] != mask_gray.shape[:2]:
            mask_gray = cv2.resize(
                mask_gray,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Convert to binary (mask > 0)
        binary_mask = to_binary_mask(mask_gray)
        soft_mask = mask_gray.copy()  # Keep for boundary losses

        # Apply paired transforms
        if self.transforms:
            transformed = self.transforms(image=image, mask=binary_mask, soft_mask=soft_mask)
            image = transformed["image"]
            binary_mask = transformed["mask"]
            soft_mask = transformed["soft_mask"]
        else:
            # Convert to tensors manually if no transforms
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            binary_mask = torch.from_numpy(binary_mask).unsqueeze(0).float()
            soft_mask = torch.from_numpy(soft_mask).unsqueeze(0).float() / 255.0

        # Ensure mask has channel dim (1, H, W)
        if binary_mask.dim() == 2:
            binary_mask = binary_mask.unsqueeze(0)
        if soft_mask.dim() == 2:
            soft_mask = soft_mask.unsqueeze(0)

        # Clone tensors for proper storage ownership
        binary_mask = binary_mask.float().contiguous().clone()
        soft_mask = soft_mask.float().contiguous().clone()
        return {
            "image": image.contiguous().clone(),
            "binary_mask": binary_mask,
            "soft_mask": soft_mask,
            "mask": binary_mask,  # Alias for notebooks expecting 'mask'
        }


class BusUcClsDataset(Dataset):
    """Dataset for breast ultrasound classification task (benign vs malignant)."""

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],  # 0=benign, 1=malignant
        transforms=None,
    ):
        """
        Args:
            image_paths: List of paths to ultrasound images
            labels: List of class labels (0 or 1)
            transforms: Albumentations transforms
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys: 'image', 'label'
        """
        # Load image
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        label = self.labels[idx]

        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]
        else:
            # Convert to tensor manually if no transforms
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }


class BusUcSegDataModule(L.LightningDataModule):
    """Lightning DataModule for segmentation task with 5-fold CV."""

    def __init__(
        self,
        data_dir: str,
        img_size: int = 256,
        batch_size: int = 16,
        num_workers: int = 4,
        augment_level: str = "medium",
        fold: int = 0,
        n_folds: int = 5,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = _resolve_repo_path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment_level = augment_level
        self.fold = fold
        self.n_folds = n_folds
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        """Split data into train/val/test or create CV folds."""
        # Get all image/mask pairs
        paths = get_data_paths(self.data_dir, task="segmentation")

        # Combine benign and malignant
        all_images = paths["benign"]["images"] + paths["malignant"]["images"]
        all_masks = paths["benign"]["masks"] + paths["malignant"]["masks"]

        # Load CV splits from JSON
        splits_path = self.data_dir.parent / "data_splits.json"
        if splits_path.exists():
            with open(splits_path, "r") as f:
                splits_data = json.load(f)

            fold_key = f"fold_{self.fold}"
            train_idx = splits_data["folds"][fold_key]["train_indices"]
            val_idx = splits_data["folds"][fold_key]["val_indices"]
            test_idx = splits_data["folds"][fold_key]["test_indices"]

            # Create datasets
            train_images = [all_images[i] for i in train_idx]
            train_masks = [all_masks[i] for i in train_idx]
            val_images = [all_images[i] for i in val_idx]
            val_masks = [all_masks[i] for i in val_idx]
            test_images = [all_images[i] for i in test_idx]
            test_masks = [all_masks[i] for i in test_idx]

            # Get transforms
            train_transforms = get_train_transforms(self.img_size, self.augment_level)
            val_transforms = get_val_transforms(self.img_size)

            self.train_dataset = BusUcSegDataset(train_images, train_masks, train_transforms)
            self.val_dataset = BusUcSegDataset(val_images, val_masks, val_transforms)
            self.test_dataset = BusUcSegDataset(test_images, test_masks, val_transforms)
        else:
            raise FileNotFoundError(f"Splits file not found: {splits_path}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


class BusUcClsDataModule(L.LightningDataModule):
    """Lightning DataModule for classification task with 5-fold CV."""

    def __init__(
        self,
        data_dir: str,
        img_size: int = 256,
        batch_size: int = 32,
        num_workers: int = 4,
        augment_level: str = "medium",
        fold: int = 0,
        n_folds: int = 5,
        use_weighted_sampler: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = _resolve_repo_path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment_level = augment_level
        self.fold = fold
        self.n_folds = n_folds
        self.use_weighted_sampler = use_weighted_sampler
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        """Split data into train/val/test or create CV folds."""
        # Get all images
        paths = get_data_paths(self.data_dir, task="classification")

        # Combine benign (0) and malignant (1)
        all_images = paths["benign"]["images"] + paths["malignant"]["images"]
        all_labels = [0] * len(paths["benign"]["images"]) + [1] * len(paths["malignant"]["images"])

        # Load CV splits from JSON
        # Note: Using same splits as segmentation task
        splits_path = self.data_dir.parent / "data_splits.json"
        if splits_path.exists():
            with open(splits_path, "r") as f:
                splits_data = json.load(f)

            fold_key = f"fold_{self.fold}"
            train_idx = splits_data["folds"][fold_key]["train_indices"]
            val_idx = splits_data["folds"][fold_key]["val_indices"]
            test_idx = splits_data["folds"][fold_key]["test_indices"]

            # Create datasets
            train_images = [all_images[i] for i in train_idx]
            train_labels = [all_labels[i] for i in train_idx]
            val_images = [all_images[i] for i in val_idx]
            val_labels = [all_labels[i] for i in val_idx]
            test_images = [all_images[i] for i in test_idx]
            test_labels = [all_labels[i] for i in test_idx]

            # Compute class weights for WeightedRandomSampler
            if self.use_weighted_sampler:
                class_counts = np.bincount(train_labels)
                class_weights = 1.0 / class_counts
                sample_weights = [class_weights[label] for label in train_labels]
                self.sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True,
                )
            else:
                self.sampler = None

            # Get transforms
            train_transforms = get_train_transforms(self.img_size, self.augment_level)
            val_transforms = get_val_transforms(self.img_size)

            self.train_dataset = BusUcClsDataset(train_images, train_labels, train_transforms)
            self.val_dataset = BusUcClsDataset(val_images, val_labels, val_transforms)
            self.test_dataset = BusUcClsDataset(test_images, test_labels, val_transforms)
        else:
            raise FileNotFoundError(f"Splits file not found: {splits_path}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
