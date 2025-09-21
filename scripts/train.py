"""Training script for biomedical image segmentation models."""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import ResUNet, UNet
from utils.data_loader import get_data_loaders
from utils.helpers import get_device, load_config, set_seed
from utils.metrics import calculate_all_metrics, dice_coefficient
from utils.visualization import learning_curve_plot, plot_predictions


def train_model(model, train_loader, val_loader, config, device):
    """Training function with comprehensive logging"""
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    # Training tracking
    train_losses, val_losses = [], []
    train_dice, val_dice = [], []
    best_dice = 0.0

    # Tensorboard writer
    writer = SummaryWriter(
        log_dir=Path(config["logging"]["log_dir"])
        / config["logging"]["experiment_name"]
    )

    start_time = time.time()

    for epoch in range(config["training"]["epochs"]):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_train_dice = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Calculate metrics
            pred = torch.sigmoid(output)
            dice = dice_coefficient((pred > 0.5).float(), target)

            epoch_train_loss += loss.item()
            epoch_train_dice += dice.item()

            # Log batch metrics
            if batch_idx % 10 == 0:
                writer.add_scalar(
                    "train/batch_loss",
                    loss.item(),
                    epoch * len(train_loader) + batch_idx,
                )
                writer.add_scalar(
                    "train/batch_dice",
                    dice.item(),
                    epoch * len(train_loader) + batch_idx,
                )

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_dice = 0.0
        all_val_metrics = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                pred = torch.sigmoid(output)
                metrics = calculate_all_metrics(pred, target)

                epoch_val_loss += loss.item()
                epoch_val_dice += metrics["dice"]
                all_val_metrics.append(metrics)

        # Calculate averages
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_dice = epoch_train_dice / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_dice = epoch_val_dice / len(val_loader)

        # Update learning rate
        scheduler.step(avg_val_dice)

        # Save best model
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(
                model.state_dict(),
                f"checkpoints/best_model_{config['model']['name']}.pth",
            )

        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_dice.append(avg_train_dice)
        val_dice.append(avg_val_dice)

        # Log epoch metrics
        writer.add_scalar("train/epoch_loss", avg_train_loss, epoch)
        writer.add_scalar("train/epoch_dice", avg_train_dice, epoch)
        writer.add_scalar("val/epoch_loss", avg_val_loss, epoch)
        writer.add_scalar("val/epoch_dice", avg_val_dice, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        print(
            f'Epoch {epoch + 1}/{config["training"]["epochs"]}: '
            f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}"
        )

    training_time = time.time() - start_time

    # Generate learning curve plot
    learning_curve_plot(
        title=f"{config['model']['name']} Learning Curves",
        train_losses=train_losses,
        test_losses=val_losses,
        train_accuracy=train_dice,
        test_accuracy=val_dice,
        batch_size=config["data"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        training_time_seconds=training_time,
    )

    # Save final model
    torch.save(
        model.state_dict(), f"checkpoints/final_model_{config['model']['name']}.pth"
    )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_dice": train_dice,
        "val_dice": val_dice,
        "best_val_dice": best_dice,
        "training_time": training_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Train biomedical segmentation model")
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--model", type=str, choices=["unet", "resunet"], help="Model architecture"
    )
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    if args.model:
        config["model"]["name"] = args.model
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    # Set seed for reproducibility
    set_seed(42)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Create data loaders
    train_loader, val_loader = get_data_loaders(config)

    # Initialize model
    if config["model"]["name"] == "unet":
        model = UNet(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            features=config["model"]["features"],
            dropout=config["model"]["dropout"],
        )
    elif config["model"]["name"] == "resunet":
        model = ResUNet(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            features=config["model"]["features"],
            dropout=config["model"]["dropout"],
        )
    else:
        raise ValueError(f"Unknown model: {config['model']['name']}")

    print(f"Model: {config['model']['name']}")
    print(f"Number of parameters: {model.count_parameters():,}")

    # Train model
    results = train_model(model, train_loader, val_loader, config, device)

    print(f"Training completed in {results['training_time']:.2f} seconds")
    print(f"Best validation Dice: {results['best_val_dice']:.4f}")

    # Visualize some predictions
    model.eval()
    with torch.no_grad():
        sample_data, sample_target = next(iter(val_loader))
        sample_data, sample_target = sample_data.to(device), sample_target.to(device)
        sample_pred = model(sample_data)

        fig = plot_predictions(sample_data, sample_target, sample_pred, num_samples=3)
        fig.savefig(f"results/{config['model']['name']}_predictions.png")


if __name__ == "__main__":
    main()
