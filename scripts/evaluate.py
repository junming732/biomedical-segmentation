"""Evaluation script for biomedical image segmentation models."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from models import ResUNet, UNet
from utils.data_loader import get_data_loaders
from utils.helpers import get_device, load_config, set_seed
from utils.metrics import calculate_all_metrics


def evaluate_model(model, test_loader, device, threshold=0.5):
    """Evaluate model on test set"""
    model.eval()
    all_metrics = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = torch.sigmoid(output)

            # Calculate metrics
            metrics = calculate_all_metrics(pred, target, threshold)
            all_metrics.append(metrics)

            # Store for overall analysis
            all_predictions.append((pred > threshold).float().cpu().numpy())
            all_targets.append(target.cpu().numpy())

    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    # Flatten all predictions and targets
    all_predictions = np.concatenate([p.flatten() for p in all_predictions])
    all_targets = np.concatenate([t.flatten() for t in all_targets])

    return avg_metrics, all_predictions, all_targets


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Background", "Gland"],
        yticklabels=["Background", "Gland"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_confusion_matrix.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate biomedical segmentation model"
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["unet", "resunet"],
        required=True,
        help="Model architecture",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Prediction threshold"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set seed for reproducibility
    set_seed(42)

    # Get device
    device = get_device()

    # Initialize model
    if args.model_type == "unet":
        model = UNet(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            features=config["model"]["features"],
            dropout=config["model"]["dropout"],
        )
    elif args.model_type == "resunet":
        model = ResUNet(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            features=config["model"]["features"],
            dropout=config["model"]["dropout"],
        )

    # Load trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # Get test data loader
    _, test_loader = get_data_loaders(config)

    # Evaluate model
    metrics, predictions, targets = evaluate_model(
        model, test_loader, device, args.threshold
    )

    # Print results
    print(f"\nEvaluation Results for {args.model_type}:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(targets, predictions, args.model_type)

    # Classification report
    print("\nClassification Report:")
    print("=" * 50)
    print(
        classification_report(
            targets, predictions, target_names=["Background", "Gland"], digits=4
        )
    )


if __name__ == "__main__":
    main()
