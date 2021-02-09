"""
 Copyright 2020 Mahmoud Afifi.
 Released under the MIT License.
 If you use this code, please cite the following paper:
 Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith
 Punnappurath, and Michael S Brown.
 CIE XYZ Net: Unprocessing Images for Low-Level Computer Vision Tasks.
 arXiv preprint, 2020.
"""

__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import torch
import torch.nn as nn
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class globalSubNet(nn.Module):
    def __init__(self, blockDepth=5, convDepth=64, in_img_sz=128,
                 device='cuda'):
        super(globalSubNet, self).__init__()
        self.blockDepth = blockDepth
        self.convDepth = convDepth
        self.in_img_sz = in_img_sz
        self.device = device
        self.net = torch.nn.Sequential()
        for i in range(self.blockDepth):
            if i == 0:
                conv = torch.nn.Conv2d(3, self.convDepth, 3, padding=1)
                torch.nn.init.kaiming_normal_(conv.weight)
                torch.nn.init.zeros_(conv.bias)

            else:
                conv = torch.nn.Conv2d(self.convDepth, self.convDepth, 3,
                                       padding=1)
                torch.nn.init.kaiming_normal_(conv.weight)
                torch.nn.init.zeros_(conv.bias)
            self.net.add_module('conv%d' % i, conv)
            self.net.add_module('leakyRelu%d' % i, torch.nn.LeakyReLU(
                inplace=False))
            self.net.add_module('maxpool%d' % i,
                                torch.nn.MaxPool2d(2, stride=2))

        self.net.add_module('flatten', Flatten())
        self.net.add_module('fc1', torch.nn.Linear(1024, 1024))
        self.net.add_module('fc1', torch.nn.Linear(1024, 1024))
        self.net.add_module('dropout', torch.nn.Dropout(p=0.5))
        self.net.add_module('out', torch.nn.Linear(1024, 3 * 6))

    def forward(self, x):
        inds_1 = torch.LongTensor(
            np.linspace(0, x.size(2), self.in_img_sz,
                        endpoint=False)).to(device=self.device)
        inds_2 = torch.LongTensor(
            np.linspace(0, x.size(3), self.in_img_sz, endpoint=False)).to(
            device=self.device)
        x_sampled = x.index_select(2, inds_1)
        x_sampled = x_sampled.index_select(3, inds_2)
        m = self.net(x_sampled)
        return m
