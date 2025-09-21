import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .metrics import dice_coefficient


def learning_curve_plot(
    title,
    train_losses,
    test_losses,
    train_accuracy,
    test_accuracy,
    batch_size,
    learning_rate,
    training_time_seconds,
):
    """Plot learning curves for training and validation"""
    fsizes = [13, 10, 8]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, y=1.1, fontsize=fsizes[0])

    # Format training time
    hours = training_time_seconds // 3600
    minutes = (training_time_seconds % 3600) // 60
    formatted_time = f"{hours}h {minutes}min"

    # Add subtitle with training info
    sub = f"Batch size: {batch_size} | Learning rate: {learning_rate} | "
    sub += f"Epochs: {len(train_losses)} | Training time: {formatted_time}"
    fig.text(0.5, 0.9, sub, ha="center", fontsize=fsizes[1])

    x = range(1, len(train_losses) + 1)

    # Plot losses
    axs[0].plot(x, train_losses, label=f"Final train loss: {train_losses[-1]:.4f}")
    axs[0].plot(x, test_losses, label=f"Final test loss: {test_losses[-1]:.4f}")
    axs[0].set_title("Losses", fontsize=fsizes[1])
    axs[0].set_xlabel("Epoch", fontsize=fsizes[1])
    axs[0].set_ylabel("Loss", fontsize=fsizes[1])
    axs[0].legend(fontsize=fsizes[2])
    axs[0].tick_params(axis="both", labelsize=fsizes[2])

    # Plot accuracy/Dice score
    axs[1].plot(
        x, train_accuracy, label=f"Final train accuracy: {train_accuracy[-1]:.4f}%"
    )
    axs[1].plot(
        x, test_accuracy, label=f"Final test accuracy: {test_accuracy[-1]:.4f}%"
    )
    axs[1].set_title("Dice Score", fontsize=fsizes[1])
    axs[1].set_xlabel("Epoch", fontsize=fsizes[1])
    axs[1].set_ylabel("Dice Score", fontsize=fsizes[1])
    axs[1].legend(fontsize=fsizes[2])
    axs[1].tick_params(axis="both", labelsize=fsizes[2])

    plt.tight_layout()
    return fig


def plot_predictions(images, masks, predictions, threshold=0.5, num_samples=3):
    """Plot sample predictions alongside ground truth"""
    fig, axes = plt.subplots(num_samples, 4, figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        # Input image (only first two channels)
        img = images[i][:2].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())

        # Ground truth mask
        mask = masks[i].cpu().numpy().squeeze()

        # Prediction
        pred = predictions[i].cpu().numpy().squeeze()
        pred_binary = (pred > threshold).astype(np.float32)

        # Plot input image
        axes[i, 0].imshow(img[:, :, 0], cmap="gray")
        axes[i, 0].set_title("Input Image (Channel 1)")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img[:, :, 1], cmap="gray")
        axes[i, 1].set_title("Input Image (Channel 2)")
        axes[i, 1].axis("off")

        # Plot ground truth
        axes[i, 2].imshow(mask, cmap="gray")
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].axis("off")

        # Plot prediction
        axes[i, 3].imshow(pred_binary, cmap="gray")
        axes[i, 3].set_title(
            f"Prediction (Dice: {dice_coefficient(pred_binary, mask):.3f})"
        )
        axes[i, 3].axis("off")

    plt.tight_layout()
    return fig


def plot_confusion_matrix(predictions, targets, threshold=0.5):
    """Plot confusion matrix for segmentation results"""
    pred_flat = (predictions > threshold).float().view(-1).cpu().numpy()
    target_flat = targets.view(-1).cpu().numpy()

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(target_flat, pred_flat)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax
