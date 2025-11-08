"""Lightning DataModules and Datasets for BUS_UC."""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import lightning as L
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Tuple, List


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
            dict with keys: 'image', 'binary_mask', 'soft_mask'
        """
        # TODO: Implement
        # - Load image and mask
        # - Convert mask to binary (mask > 0)
        # - Keep soft_mask as grayscale for boundary losses
        # - Apply transforms (paired for image+mask)
        # - Convert to tensors
        raise NotImplementedError


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
        # TODO: Implement
        # - Load image
        # - Apply transforms
        # - Convert to tensor
        raise NotImplementedError


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
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment_level = augment_level
        self.fold = fold
        self.n_folds = n_folds
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        """Split data into train/val/test or create CV folds."""
        # TODO: Implement
        # - Discover all image/mask pairs in BUS_UC/BUS_UC/BUS_UC/
        # - Create stratified splits by class × lesion area quintile
        # - Assign to train_dataset, val_dataset, test_dataset
        raise NotImplementedError

    def train_dataloader(self):
        # TODO: Return DataLoader with train_dataset
        raise NotImplementedError

    def val_dataloader(self):
        # TODO: Return DataLoader with val_dataset
        raise NotImplementedError

    def test_dataloader(self):
        # TODO: Return DataLoader with test_dataset
        raise NotImplementedError


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
        self.data_dir = Path(data_dir)
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
        # TODO: Implement
        # - Discover all images in BUS_UC_classification/BUS_UC_classification/
        # - Create stratified splits by class
        # - Compute class weights for WeightedRandomSampler
        raise NotImplementedError

    def train_dataloader(self):
        # TODO: Return DataLoader with WeightedRandomSampler if enabled
        raise NotImplementedError

    def val_dataloader(self):
        # TODO: Return DataLoader with val_dataset
        raise NotImplementedError

    def test_dataloader(self):
        # TODO: Return DataLoader with test_dataset
        raise NotImplementedError
