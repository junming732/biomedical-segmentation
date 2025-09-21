import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(config_path):
    """Load configuration from YAML file

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed=42):
    """Set random seed for reproducibility

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the available device (CUDA if available, else CPU)

    Returns:
        torch.device: Available device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_config(config, path):
    """Save configuration to YAML file

    Args:
        config (dict): Configuration dictionary
        path (str): Path to save the configuration file
    """
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def create_directories(config):
    """Create necessary directories for the project

    Args:
        config (dict): Configuration dictionary
    """
    directories = [
        config["logging"]["checkpoint_dir"],
        config["logging"]["log_dir"],
        "results",
        "data/raw",
        "data/processed",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def count_parameters(model):
    """Count the number of trainable parameters in a model

    Args:
        model (torch.nn.Module): PyTorch model

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model, input_size=(2, 128, 128)):
    """Get a summary of the model architecture

    Args:
        model (torch.nn.Module): PyTorch model
        input_size (tuple): Input tensor size

    Returns:
        str: Model summary
    """
    import contextlib
    import io

    from torchsummary import summary

    # Capture the summary output
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        summary(model, input_size=input_size, device="cpu")

    return f.getvalue()


def setup_experiment(config, experiment_name):
    """Set up experiment directory and configuration

    Args:
        config (dict): Base configuration
        experiment_name (str): Name of the experiment

    Returns:
        tuple: (updated_config, experiment_path)
    """
    # Create experiment directory
    experiment_path = Path("experiments") / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    # Update config with experiment name
    config["logging"]["experiment_name"] = experiment_name
    config["logging"]["log_dir"] = str(experiment_path / "logs")
    config["logging"]["checkpoint_dir"] = str(experiment_path / "checkpoints")

    # Save experiment config
    save_config(config, experiment_path / "config.yaml")

    # Create directories
    create_directories(config)

    return config, experiment_path


def load_model(model, checkpoint_path, device="cpu"):
    """Load model weights from checkpoint

    Args:
        model (torch.nn.Module): Model instance
        checkpoint_path (str): Path to checkpoint file
        device (str): Device to load model on

    Returns:
        torch.nn.Module: Model with loaded weights
    """
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def get_optimizer(model, config):
    """Get optimizer based on configuration

    Args:
        model (torch.nn.Module): Model to optimize
        config (dict): Configuration dictionary

    Returns:
        torch.optim.Optimizer: Optimizer instance
    """
    optimizer_name = config["training"]["optimizer"].lower()
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, config):
    """Get learning rate scheduler based on configuration

    Args:
        optimizer (torch.optim.Optimizer): Optimizer instance
        config (dict): Configuration dictionary

    Returns:
        torch.optim.lr_scheduler: Scheduler instance
    """
    scheduler_name = config["training"]["scheduler"].lower()

    if scheduler_name == "reduce_lr_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["training"]["epochs"]
        )
    else:
        return None


def get_loss_function(config):
    """Get loss function based on configuration

    Args:
        config (dict): Configuration dictionary

    Returns:
        torch.nn.Module: Loss function
    """
    loss_name = config["training"]["loss"].lower()

    if loss_name == "bce":
        return torch.nn.BCEWithLogitsLoss()
    elif loss_name == "dice":
        # You would need to implement Dice loss here
        return torch.nn.BCEWithLogitsLoss()  # Placeholder
    elif loss_name == "focal":
        # You would need to implement Focal loss here
        return torch.nn.BCEWithLogitsLoss()  # Placeholder
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def calculate_flops(model, input_size=(2, 128, 128)):
    """Calculate FLOPs for the model

    Args:
        model (torch.nn.Module): PyTorch model
        input_size (tuple): Input tensor size

    Returns:
        float: FLOPs count
    """
    from thop import profile

    input_tensor = torch.randn(1, *input_size)
    flops, _ = profile(model, inputs=(input_tensor,))
    return flops


def model_size_mb(model):
    """Calculate model size in megabytes

    Args:
        model (torch.nn.Module): PyTorch model

    Returns:
        float: Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def time_inference(model, input_size=(2, 128, 128), device="cpu", num_runs=100):
    """Time model inference

    Args:
        model (torch.nn.Module): PyTorch model
        input_size (tuple): Input tensor size
        device (str): Device to run on
        num_runs (int): Number of runs for timing

    Returns:
        float: Average inference time in milliseconds
    """
    model.to(device)
    model.eval()

    input_tensor = torch.randn(1, *input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # Timing
    if device == "cuda":
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
    else:
        start_time = None
        end_time = None

    if device == "cuda":
        torch.cuda.synchronize()
        start_time.record()
    else:
        start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)

    if device == "cuda":
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / num_runs
    else:
        elapsed_time = (time.time() - start_time) * 1000 / num_runs

    return elapsed_time


def setup_logging(config):
    """Set up logging for the experiment

    Args:
        config (dict): Configuration dictionary

    Returns:
        tuple: (logger, writer) for file logging and tensorboard
    """
    import logging

    from torch.utils.tensorboard import SummaryWriter

    # Create experiment directory
    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up file logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "experiment.log"),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger(__name__)

    # Set up tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    return logger, writer
