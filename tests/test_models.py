import torch

from models.resunet import ResUNet
from models.unet import UNet


def test_unet_initialization():
    """Test UNet model initialization and forward pass"""
    model = UNet(in_channels=2, out_channels=1)

    # Test model parameters
    assert hasattr(model, "inc")
    assert hasattr(model, "down1")
    assert hasattr(model, "up1")
    assert hasattr(model, "outc")

    # Test forward pass with dummy data
    dummy_input = torch.randn(2, 2, 128, 128)
    output = model(dummy_input)

    assert output.shape == (2, 1, 128, 128)
    assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output


def test_resunet_initialization():
    """Test ResUNet model initialization"""
    model = ResUNet(in_channels=2, out_channels=1)

    dummy_input = torch.randn(2, 2, 128, 128)
    output = model(dummy_input)

    assert output.shape == (2, 1, 128, 128)


def test_model_parameter_count():
    """Test that models have reasonable parameter counts"""
    unet = UNet()
    resunet = ResUNet()

    unet_params = unet.count_parameters()
    resunet_params = resunet.count_parameters()

    # Both should have parameters in the millions
    assert unet_params > 1e6
    assert resunet_params > 1e6

    # ResUNet should have more parameters due to residual connections
    assert resunet_params > unet_params


def test_model_save_load():
    """Test model saving and loading"""
    model = UNet()
    dummy_input = torch.randn(1, 2, 128, 128)

    # Save model
    model.save("test_model.pth")

    # Create new model and load weights
    new_model = UNet()
    new_model.load("test_model.pth")

    # Check outputs are the same
    output1 = model(dummy_input)
    output2 = new_model(dummy_input)

    assert torch.allclose(output1, output2, atol=1e-6)
