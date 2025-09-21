import glob
import math
import os

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class WarwickDataset(Dataset):
    """Warwick Biomedical Image Segmentation Dataset"""

    def __init__(self, image_dir, mask_dir, transform=None, split="train"):
        """Initialize the Warwick dataset."""
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.image_files = self.image_paths  # for tests
        self.mask_files = self.mask_paths
        self.transform = transform
        self.split = split

        # Split dataset 80/20
        n = len(self.image_paths)
        split_idx = math.floor(0.8 * n)

        if split == "train":
            self.image_paths = self.image_paths[:split_idx]
            self.mask_paths = self.mask_paths[:split_idx]
        elif split == "val":
            self.image_paths = self.image_paths[split_idx:]
            self.mask_paths = self.mask_paths[split_idx:]

        self.image_files = self.image_paths
        self.mask_files = self.mask_paths

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load an image-mask pair, apply transforms, and return as tensors."""
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        image = np.array(image)
        mask = np.array(mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # CHW
        mask = torch.tensor(mask, dtype=torch.float32)  # HxW

        return image, mask


def get_transforms(augment=False, image_size=(128, 128)):
    """Get data transformations"""
    if augment:
        return A.Compose(
            [
                A.Resize(image_size[0], image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.Normalize(mean=[0.5, 0.5, 0.0], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(mean=[0.5, 0.5, 0.0], std=[0.5, 0.5, 0.5]),
            ]
        )


def get_data_loaders(config):
    """Get data loaders for training and validation"""
    train_transform = get_transforms(augment=config["data"]["augmentations"])
    val_transform = get_transforms(augment=False)

    # Create datasets
    train_dataset = WarwickDataset(
        os.path.join(config["data"]["data_dir"], "train", "images"),
        os.path.join(config["data"]["data_dir"], "train", "masks"),
        transform=train_transform,
        split="train",
    )

    val_dataset = WarwickDataset(
        os.path.join(config["data"]["data_dir"], "val", "images"),
        os.path.join(config["data"]["data_dir"], "val", "masks"),
        transform=val_transform,
        split="val",
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader
