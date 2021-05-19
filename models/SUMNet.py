#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 22:57:38 2018

@author: sumanthnandamuri
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torchvision import models

class SUMNet(nn.Module):
    def __init__(self):
        super(SUMNet, self).__init__()
        
        self.encoder   = models.vgg11(pretrained = True).features
        self.preconv   = nn.Conv2d(3, 3, 1)
        self.conv1     = self.encoder[0]
        self.pool1     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv2     = self.encoder[3]
        self.pool2     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv3a    = self.encoder[6]
        self.conv3b    = self.encoder[8]
        self.pool3     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv4a    = self.encoder[11]
        self.conv4b    = self.encoder[13]
        self.pool4     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv5a    = self.encoder[16]
        self.conv5b    = self.encoder[18]
        self.pool5     = nn.MaxPool2d(2, 2, return_indices = True)
        
        self.unpool5   = nn.MaxUnpool2d(2, 2)
        self.donv5b    = nn.Conv2d(1024, 512, 3, padding = 1)
        self.donv5a    = nn.Conv2d(512, 512, 3, padding = 1)
        self.unpool4   = nn.MaxUnpool2d(2, 2)
        self.donv4b    = nn.Conv2d(1024, 512, 3, padding = 1)
        self.donv4a    = nn.Conv2d(512, 256, 3, padding = 1)
        self.unpool3   = nn.MaxUnpool2d(2, 2)
        self.donv3b    = nn.Conv2d(512, 256, 3, padding = 1)
        self.donv3a    = nn.Conv2d(256,128, 3, padding = 1)
        self.unpool2   = nn.MaxUnpool2d(2, 2)
        self.donv2     = nn.Conv2d(256, 64, 3, padding = 1)
        self.unpool1   = nn.MaxUnpool2d(2, 2)
        self.donv1     = nn.Conv2d(128, 32, 3, padding = 1)
        self.output    = nn.Conv2d(32, 5, 1)
        
    def forward(self, x):
        preconv        = F.relu(self.preconv(x), inplace = True)
        conv1          = F.relu(self.conv1(preconv), inplace = True)
        pool1, idxs1   = self.pool1(conv1)
        conv2          = F.relu(self.conv2(pool1), inplace = True)
        pool2, idxs2   = self.pool2(conv2)
        conv3a         = F.relu(self.conv3a(pool2), inplace = True)
        conv3b         = F.relu(self.conv3b(conv3a), inplace = True)
        pool3, idxs3   = self.pool3(conv3b)
        conv4a         = F.relu(self.conv4a(pool3), inplace = True)
        conv4b         = F.relu(self.conv4b(conv4a), inplace = True)
        pool4, idxs4   = self.pool4(conv4b)
        conv5a         = F.relu(self.conv5a(pool4), inplace = True)
        conv5b         = F.relu(self.conv5b(conv5a), inplace = True)
        pool5, idxs5   = self.pool5(conv5b)
        
        unpool5        = torch.cat([self.unpool5(pool5, idxs5), conv5b], 1)
        donv5b         = F.relu(self.donv5b(unpool5), inplace = True)
        donv5a         = F.relu(self.donv5a(donv5b), inplace = True)
        unpool4        = torch.cat([self.unpool4(donv5a, idxs4), conv4b], 1)
        donv4b         = F.relu(self.donv4b(unpool4), inplace = True)
        donv4a         = F.relu(self.donv4a(donv4b), inplace = True)
        unpool3        = torch.cat([self.unpool3(donv4a, idxs3), conv3b], 1)
        donv3b         = F.relu(self.donv3b(unpool3), inplace = True)
        donv3a         = F.relu(self.donv3a(donv3b))
        unpool2        = torch.cat([self.unpool2(donv3a, idxs2), conv2], 1)
        donv2          = F.relu(self.donv2(unpool2), inplace = True)
        unpool1        = torch.cat([self.unpool1(donv2, idxs1), conv1], 1)
        donv1          = F.relu(self.donv1(unpool1), inplace = True)
        output         = self.output(donv1)
        return torch.sigmoid(output)