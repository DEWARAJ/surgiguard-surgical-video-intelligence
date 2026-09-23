from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvGRUCell(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gates = nn.Conv2d(channels * 2, channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(channels * 2, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None) -> torch.Tensor:
        if hidden is None:
            hidden = torch.zeros_like(x)
        reset, update = torch.sigmoid(self.gates(torch.cat([x, hidden], dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * hidden], dim=1)))
        return (1.0 - update) * hidden + update * candidate


class TemporalSurgiNet(nn.Module):
    """Compact segmentation + tip-localization network with temporal memory."""

    def __init__(self, base_channels: int = 8, classes: int = 3):
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock(3, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.bottleneck = ConvBlock(b * 2, b * 4)
        self.pool = nn.MaxPool2d(2)
        self.temporal = ConvGRUCell(b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)
        self.segmentation_head = nn.Conv2d(b, classes, 1)
        self.tip_head = nn.Sequential(nn.Conv2d(b, 1, 1), nn.Sigmoid())

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if video.ndim != 5:
            raise ValueError("expected video shape [batch, time, channels, height, width]")
        hidden = None
        segmentations, tips = [], []
        for frame in video.unbind(dim=1):
            skip1 = self.enc1(frame)
            skip2 = self.enc2(self.pool(skip1))
            encoded = self.bottleneck(self.pool(skip2))
            hidden = self.temporal(encoded, hidden)
            decoded2 = self.dec2(torch.cat([self.up2(hidden), skip2], dim=1))
            decoded1 = self.dec1(torch.cat([self.up1(decoded2), skip1], dim=1))
            segmentations.append(self.segmentation_head(decoded1))
            tips.append(self.tip_head(decoded1))
        return torch.stack(segmentations, dim=1), torch.stack(tips, dim=1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

