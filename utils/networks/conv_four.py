# -*- coding: utf-8 -*-
"""
Adapted from: https://github.com/wyharveychen/CloserLookFewShot
This file contains Conv32F(ReLU/LeakyReLU), Conv64F(ReLU/LeakyReLU) and R2D2Embedding.
"""

import torch
import torch.nn as nn


class Conv64F(nn.Module):
    """
    Four convolutional blocks network, each of which consists of a Covolutional layer,
    a Batch Normalizaiton layer, a ReLU layer and a Maxpooling layer.
    Used in the original ProtoNet: https://github.com/jakesnell/prototypical-networks.git.

    Input:  3 * 84 *84
    Output: 64 * 5 * 5
    """

    def __init__(
            self,
            is_flatten=True,
            is_feature=False,
            leaky_relu=False,
            negative_slope=0.2,
            last_pool=True,
            maxpool_last2=True,
    ):
        super(Conv64F, self).__init__()

        self.is_flatten = is_flatten
        self.is_feature = is_feature
        self.last_pool = last_pool
        self.maxpool_last2 = maxpool_last2

        if leaky_relu:
            activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        else:
            activation = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            activation,
        )
        self.layer3_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            activation,
        )
        self.layer4_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if is_flatten:
            self.final_feat_dim = 1600
        else:
            self.final_feat_dim = (64, 5, 5)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        if self.maxpool_last2:
            out3 = self.layer3_maxpool(out3)  # for some methods(relation net etc.)

        out4 = self.layer4(out3)
        if self.last_pool:
            out4 = self.layer4_pool(out4)

        if self.is_flatten:
            out4 = out4.view(out4.size(0), -1)

        if self.is_feature:
            return out1, out2, out3, out4

        return out4


class Conv32F(nn.Module):
    """
    Four convolutional blocks network, each of which consists of a Covolutional layer,
    a Batch Normalizaiton layer, a ReLU layer and a Maxpooling layer.
    Used in the original ProtoNet: https://github.com/jakesnell/prototypical-networks.git.

    Input:  3 * 84 *84
    Output: 32 * 5 * 5
    """

    def __init__(
            self,
            is_flatten=True,
            is_feature=False,
            leaky_relu=False,
            negative_slope=0.2,
            last_pool=True,
    ):
        super(Conv32F, self).__init__()

        self.is_flatten = is_flatten
        self.is_feature = is_feature
        self.last_pool = last_pool

        if leaky_relu:
            activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        else:
            activation = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            activation,
        )
        self.layer4_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if is_flatten:
            self.final_feat_dim = 800
        else:
            self.final_feat_dim = (32, 5, 5)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        if self.last_pool:
            out4 = self.layer4_pool(out4)

        if self.is_flatten:
            out4 = out4.view(out4.size(0), -1)

        if self.is_feature:
            return out1, out2, out3, out4

        return out4