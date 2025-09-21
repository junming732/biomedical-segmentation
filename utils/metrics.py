def dice_coefficient(pred, target, smooth=1.0):
    """Calculate Dice coefficient between prediction and target

    Args:
        pred: predicted segmentation mask (after thresholding)
        target: ground truth mask
        smooth: smoothing factor to avoid division by zero

    Returns:
        dice_score: Dice coefficient value
    """
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )

    return dice


def iou_score(pred, target, smooth=1.0):
    """Calculate Intersection over Union (IoU)"""
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    total = (pred_flat + target_flat).sum()
    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou


def precision_recall(pred, target, smooth=1.0):
    """Calculate precision and recall metrics"""
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()

    precision = (intersection + smooth) / (pred_flat.sum() + smooth)
    recall = (intersection + smooth) / (target_flat.sum() + smooth)

    return precision, recall


def calculate_all_metrics(pred, target, threshold=0.5):
    """Calculate all evaluation metrics"""
    pred_binary = (pred > threshold).float()

    dice = dice_coefficient(pred_binary, target)
    iou = iou_score(pred_binary, target)
    precision, recall = precision_recall(pred_binary, target)

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
    }
