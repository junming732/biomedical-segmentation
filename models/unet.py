"""UNet architecture for biomedical image segmentation."""

import torch
import torch.nn as nn

from .base_model import BaseSegmentationModel


class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, dropout=0.0, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle potential size mismatches
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(BaseSegmentationModel):
    """UNet architecture with skip connections"""

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
        self.inc = DoubleConv(in_channels, features[0], dropout)
        self.down1 = Down(features[0], features[1], dropout)
        self.down2 = Down(features[1], features[2], dropout)
        self.down3 = Down(features[2], features[3], dropout)

        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * factor, dropout)

        # Decoder
        self.up1 = Up(
            features[3] * factor + features[3], features[3], dropout, bilinear
        )
        self.up2 = Up(features[3] + features[2], features[2], dropout, bilinear)
        self.up3 = Up(features[2] + features[1], features[1], dropout, bilinear)
        self.up4 = Up(features[1] + features[0], features[0], dropout, bilinear)

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
