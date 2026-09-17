"""Tiny segmentation model for class-page foreground extraction."""

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyUNet(nn.Module):
    """Compact U-Net; output is a foreground logit map."""

    def __init__(self, base_channels: int = 16, depth: int = 3):
        super().__init__()
        channels = [base_channels * (2 ** i) for i in range(depth)]
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = 3
        for ch in channels:
            self.encoders.append(ConvBlock(in_ch, ch))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = ch

        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        dec_in = channels[-1] * 2
        for ch in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(dec_in, ch, 2, stride=2))
            self.decoders.append(ConvBlock(ch * 2, ch))
            dec_in = ch
        self.head = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.head(x)
