# -*- coding: utf-8 -*-
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.core import ResidualConv2


class CasSeg(nn.Module):
    def __init__(self, channel, filters=None):
        super(CasSeg, self).__init__()

        if filters is None:
            filters = [64, 128, 256, 512]
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv2(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv2(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv2(filters[2], filters[3], 2, 1)

        self.post_layer1 = nn.Conv2d(filters[3], filters[2], 2, 1)
        self.post_layer2 = nn.Conv2d(filters[2], filters[1], 2, 1)
        self.post_layer3 = nn.Conv2d(filters[1], filters[0], 2, 1)
        self.features = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5))
        )
        self.linear1 = nn.Linear(5 * 5 * filters[0], filters[1])
        self.linear2 = nn.Linear(filters[1], filters[0])
        self.linear3 = nn.Linear(filters[0], 32)
        self.bn1 = nn.BatchNorm1d(32, momentum=0.1)
        self.output = nn.Sequential(
            nn.Linear(32, 5),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.bridge(x3)
        x5 = self.post_layer1(x4)
        x6 = self.post_layer2(x5)
        x7 = self.post_layer3(x6)
        x8 = self.features(x7)
        x9 = torch.flatten(x8, 1)
        x10 = self.linear1(x9)
        x11 = self.linear2(x10)
        x12 = self.linear3(x11)
        x13 = self.bn1(x12)
        output = self.output(x13)
        return output
