import torch
import yaml

from utils.helpers import get_device, load_config, set_seed


def test_load_config():
    """Test configuration loading"""
    # Create a temporary config file
    config_data = {
        "model": {"name": "test", "in_channels": 3},
        "training": {"epochs": 10},
    }

    with open("test_config.yaml", "w") as f:
        yaml.dump(config_data, f)

    # Load config
    config = load_config("test_config.yaml")

    assert config["model"]["name"] == "test"
    assert config["model"]["in_channels"] == 3
    assert config["training"]["epochs"] == 10

    # Clean up
    import os

    os.remove("test_config.yaml")


def test_set_seed():
    """Test seed setting for reproducibility"""
    set_seed(42)

    # Generate some random numbers
    rand1 = torch.rand(5)
    rand2 = torch.rand(5)

    # Reset and generate again
    set_seed(42)
    rand1_again = torch.rand(5)
    rand2_again = torch.rand(5)

    # Should be identical
    assert torch.allclose(rand1, rand1_again)
    assert torch.allclose(rand2, rand2_again)


def test_get_device():
    """Test device detection"""
    device = get_device()

    # Should return a torch device
    assert isinstance(device, torch.device)

    # Should prefer CUDA if available, else CPU
    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"
