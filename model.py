import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# MobileFaceNet Architecture


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=0, groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class DepthWiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthWiseBlock, self).__init__()
        self.use_shortcut = (stride == 1 and in_channels == out_channels)

        self.conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel=3,
                      stride=stride, padding=1, groups=in_channels),
            ConvBlock(in_channels, out_channels, kernel=1, stride=1)
        )

        if not self.use_shortcut:
            self.shortcut = ConvBlock(
                in_channels, out_channels, kernel=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=512):
        super(MobileFaceNet, self).__init__()

        self.backbone = nn.Sequential(
            ConvBlock(3, 64, kernel=3, stride=2, padding=1),
            ConvBlock(64, 64, kernel=3, stride=1, padding=1, groups=64),

            DepthWiseBlock(64, 64, stride=2),
            DepthWiseBlock(64, 64, stride=1),
            DepthWiseBlock(64, 64, stride=1),
            DepthWiseBlock(64, 64, stride=1),

            DepthWiseBlock(64, 128, stride=2),
            DepthWiseBlock(128, 128, stride=1),
            DepthWiseBlock(128, 128, stride=1),
            DepthWiseBlock(128, 128, stride=1),
            DepthWiseBlock(128, 128, stride=1),
            DepthWiseBlock(128, 128, stride=1),

            ConvBlock(128, 512, kernel=1, stride=1)
        )

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.output_layer(x)
        return x
