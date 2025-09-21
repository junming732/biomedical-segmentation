"""Base model class for segmentation models."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseSegmentationModel(nn.Module, ABC):
    """Abstract base class for segmentation models"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def save(self, path):
        """Save model state dictionary"""
        torch.save(self.state_dict(), path)

    def load(self, path, device="cpu"):
        """Load model state dictionary"""
        self.load_state_dict(torch.load(path, map_location=device))
