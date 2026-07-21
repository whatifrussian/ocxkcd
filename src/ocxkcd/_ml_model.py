from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class SmallUNet(nn.Module):
    """Compact four-channel U-Net suitable for later CPU inference."""

    def __init__(self, input_channels: int = 4, base_channels: int = 16) -> None:
        super().__init__()
        self.down1 = DoubleConv(input_channels, base_channels)
        self.down2 = DoubleConv(base_channels, base_channels * 2)
        self.down3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.bridge = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = DoubleConv(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up2 = DoubleConv(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up1 = DoubleConv(base_channels * 2 + base_channels, base_channels)
        self.output = nn.Conv2d(base_channels, 1, 1)

    @staticmethod
    def upsample(value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first = self.down1(value)
        second = self.down2(self.pool(first))
        third = self.down3(self.pool(second))
        bridge = self.bridge(self.pool(third))
        value = self.up3(torch.cat((self.upsample(bridge, third), third), dim=1))
        value = self.up2(torch.cat((self.upsample(value, second), second), dim=1))
        value = self.up1(torch.cat((self.upsample(value, first), first), dim=1))
        return self.output(value)
