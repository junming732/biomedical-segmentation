import os
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from utils.data_loader import WarwickDataset, get_transforms


def create_test_dataset(tmpdir, num_samples=5):
    img_dir = os.path.join(tmpdir, "images")
    mask_dir = os.path.join(tmpdir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(num_samples):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (128, 128), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(img_dir, f"{i}.png"))
        Image.fromarray(mask).save(os.path.join(mask_dir, f"{i}.png"))

    return img_dir, mask_dir


def test_warwick_dataset_creation():
    """Test WarwickDataset creation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir, mask_dir = create_test_dataset(tmpdir)

        # Test without transforms
        dataset = WarwickDataset(img_dir, mask_dir, transform=None)

        assert len(dataset) == 4
        assert dataset.split == "train"

        # Test with transforms
        transform = get_transforms(augment=False)
        dataset = WarwickDataset(img_dir, mask_dir, transform=transform)

        image, mask = dataset[0]

        assert image.shape == (3, 128, 128)  # CHW format
        assert mask.shape == (128, 128)
        assert mask.dtype == torch.float32
        assert torch.all(mask >= 0) and torch.all(mask <= 1)


def test_data_transforms():
    """Test data transformations"""
    try:
        import numpy as np
    except ImportError:
        pytest.skip("NumPy not available")

    transform = get_transforms(augment=False)
    augment_transform = get_transforms(augment=True)

    # Create test data
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)

    # Test basic transform
    transformed = transform(image=image, mask=mask)
    assert transformed["image"].shape == (128, 128, 3)
    assert transformed["mask"].shape == (128, 128)

    # Test augmentation transform
    transformed = augment_transform(image=image, mask=mask)
    assert transformed["image"].shape == (128, 128, 3)
    assert transformed["mask"].shape == (128, 128)


def test_dataset_splitting():
    """Test dataset train/val splitting"""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir, mask_dir = create_test_dataset(tmpdir)

        train_dataset = WarwickDataset(img_dir, mask_dir, split="train")
        val_dataset = WarwickDataset(img_dir, mask_dir, split="val")

        # Should have different splits (80/20)
        assert len(train_dataset) == 4  # 80% of 5
        assert len(val_dataset) == 1  # 20% of 5

        # Should have different images
        train_files = set(train_dataset.image_files)
        val_files = set(val_dataset.image_files)

        assert len(train_files.intersection(val_files)) == 0  # No overlap
