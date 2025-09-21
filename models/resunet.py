"""Residual UNet architecture for biomedical image segmentation."""

import torch
import torch.nn as nn

from .base_model import BaseSegmentationModel


class ResidualBlock(nn.Module):
    """Residual block with two convolutions"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ResDown(nn.Module):
    """Downscaling with maxpool then residual block"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.maxpool_res = nn.Sequential(
            nn.MaxPool2d(2), ResidualBlock(in_channels, out_channels, dropout)
        )

    def forward(self, x):
        return self.maxpool_res(x)


class ResUp(nn.Module):
    """Upscaling then residual block"""

    def __init__(self, in_channels, out_channels, dropout=0.0, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.res_block = ResidualBlock(in_channels, out_channels, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle potential size mismatches
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.res_block(x)


class ResUNet(BaseSegmentationModel):
    """Residual UNet architecture"""

    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        features=[64, 128, 256, 512],
        dropout=0.0,
        bilinear=True,
    ):
        super().__init__()
        self.bilinear = bilinear

        # Encoder
        self.inc = ResidualBlock(in_channels, features[0], dropout)
        self.down1 = ResDown(features[0], features[1], dropout)
        self.down2 = ResDown(features[1], features[2], dropout)
        self.down3 = ResDown(features[2], features[3], dropout)

        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = ResDown(features[3], features[3] * factor, dropout)

        # Decoder
        self.up1 = ResUp(
            features[3] * factor + features[3], features[3], dropout, bilinear
        )
        self.up2 = ResUp(features[3] + features[2], features[2], dropout, bilinear)
        self.up3 = ResUp(features[2] + features[1], features[1], dropout, bilinear)
        self.up4 = ResUp(features[1] + features[0], features[0], dropout, bilinear)

        # Final convolution
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.initialize_weights()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Final output
        logits = self.outc(x)
        return torch.sigmoid(logits)
