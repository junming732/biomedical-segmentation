"""Prediction script for biomedical image segmentation."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from models import ResUNet, UNet
from utils.data_loader import get_transforms
from utils.helpers import get_device, load_config


def predict_single_image(model, image_path, device, transform, threshold=0.5):
    """Predict segmentation for a single image"""
    # Load and preprocess image
    image = np.array(Image.open(image_path))
    transformed = transform(image=image)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output)
        binary_mask = (prediction > threshold).float()

    return image_tensor.cpu().squeeze(), binary_mask.cpu().squeeze()


def main():
    parser = argparse.ArgumentParser(
        description="Predict segmentation for biomedical images"
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
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing images to predict",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="predictions",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Prediction threshold"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get transform
    transform = get_transforms(augment=False)

    # Process all images in directory
    image_files = [
        f for f in os.listdir(args.image_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    for image_file in image_files:
        image_path = os.path.join(args.image_dir, image_file)

        try:
            # Predict
            image_tensor, prediction = predict_single_image(
                model, image_path, device, transform, args.threshold
            )

            # Convert to numpy for visualization
            image = image_tensor[:2].permute(1, 2, 0).numpy()
            image = (image - image.min()) / (image.max() - image.min())
            prediction = prediction.numpy()

            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Original image (first channel)
            ax1.imshow(image[:, :, 0], cmap="gray")
            ax1.set_title("Input Image (Channel 1)")
            ax1.axis("off")

            # Prediction
            ax2.imshow(prediction, cmap="gray")
            ax2.set_title("Segmentation Prediction")
            ax2.axis("off")

            # Save result
            output_path = os.path.join(args.output_dir, f"pred_{image_file}")
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close()

            # Save raw prediction mask
            pred_image = Image.fromarray((prediction * 255).astype(np.uint8))
            pred_image.save(os.path.join(args.output_dir, f"mask_{image_file}"))

            print(f"Processed: {image_file}")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")


if __name__ == "__main__":
    main()
