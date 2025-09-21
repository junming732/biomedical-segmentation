import torch

from utils.metrics import (
    calculate_all_metrics,
    dice_coefficient,
    iou_score,
    precision_recall,
)


def test_dice_coefficient_perfect():
    """Test dice coefficient with perfect prediction"""
    pred = torch.ones(10, 10)
    target = torch.ones(10, 10)

    dice = dice_coefficient(pred, target)
    assert abs(dice.item() - 1.0) < 1e-6


def test_dice_coefficient_zero():
    """Test dice coefficient with zero prediction"""
    pred = torch.zeros(10, 10)
    target = torch.ones(10, 10)

    dice = dice_coefficient(pred, target, smooth=0.0)
    assert abs(dice.item() - 0.0) < 1e-6


def test_iou_score_perfect():
    """Test IoU with perfect prediction"""
    pred = torch.ones(10, 10)
    target = torch.ones(10, 10)

    iou = iou_score(pred, target)
    assert abs(iou.item() - 1.0) < 1e-6


def test_precision_recall_perfect():
    """Test precision and recall with perfect prediction"""
    pred = torch.ones(10, 10)
    target = torch.ones(10, 10)

    precision, recall = precision_recall(pred, target)
    assert abs(precision.item() - 1.0) < 1e-6
    assert abs(recall.item() - 1.0) < 1e-6


def test_calculate_all_metrics():
    """Test calculation of all metrics"""
    pred = torch.ones(10, 10)
    target = torch.ones(10, 10)

    metrics = calculate_all_metrics(pred, target)

    assert "dice" in metrics
    assert "iou" in metrics
    assert "precision" in metrics
    assert "recall" in metrics

    assert all(0 <= value <= 1 for value in metrics.values())


def test_metrics_with_threshold():
    """Test metrics with different thresholds"""
    pred = torch.tensor([[0.6, 0.4], [0.3, 0.7]])
    target = torch.tensor([[1.0, 1.0], [0.0, 1.0]])

    # Test with threshold 0.5
    metrics_05 = calculate_all_metrics(pred, target, threshold=0.5)

    # Test with threshold 0.3
    metrics_03 = calculate_all_metrics(pred, target, threshold=0.3)

    # Should be different
    assert metrics_05["dice"] != metrics_03["dice"]
