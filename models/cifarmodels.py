import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
import utils

class Model_1k(nn.Module):
    def __init__(self):
        super().__init__()
        #32x32
        self.conv1 = utils.DepthwiseSeparableConv(3, 6, (3,3), stride=2) #15x15, params: [(3+3)*3] + [3*6*1*1] = 36

        self.sobel_x = utils.SobelXConv(3, stride=2)
        self.sobel_y = utils.SobelYConv(3, stride=2)
        self.high_pass = utils.HighPassConv(3, stride=2)
        self.laplacian = utils.LaplacianConv(3, stride=2)
        self.sharpen = utils.SharpenConv(3, stride=2)
        self.batch1 = nn.BatchNorm2d(11)  # params: 11 + 11 = 22

        self.conv2 = utils.DepthwiseSeparableConv(11, 12, (3, 3), stride=2) #7x7, params: [(3+3)*11] + [11*12*1*1] = 198
        self.batch2 = nn.BatchNorm2d(12) #params: 12 + 12 = 24

        self.conv3 = utils.DepthwiseSeparableConv(12, 18, (3, 3), stride=2) #3x3, params: [(3+3)*12] + [12*18*1*1] = 288
        self.batch3 = nn.BatchNorm2d(18) #params: 18 + 18 = 36

        self.pool = nn.AvgPool2d((3,3))
        self.last = utils.LowRankClassifier(18, 100, 5) #params [18 * 5] + [5*100] = 590


    def forward(self, x):
        input = x
        sobel_x = self.sobel_x(input)
        sobel_y = self.sobel_y(input)
        high_pass = self.high_pass(input)
        laplacian = self.laplacian(input)
        sharpen = self.sharpen(input)

        x = self.conv1(input)
        added_x = torch.cat((sobel_x, sobel_y, high_pass, laplacian, sharpen, x), 1)

        x = self.batch1(added_x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)

        x = self.pool(x)
        x = self.last(x)
        return x

class Model_5k(nn.Module):
    def __init__(self):
        super().__init__()
        #32x32
        self.conv1 = utils.DepthwiseSeparableConv(3, 6, (3,3), stride=2) #15x15, params: [(3+3)*3] + [3*6*1*1] = 36

        self.sobel_x = utils.SobelXConv(3, stride=2)
        self.sobel_y = utils.SobelYConv(3, stride=2)
        self.high_pass = utils.HighPassConv(3, stride=2)
        self.laplacian = utils.LaplacianConv(3, stride=2)
        self.sharpen = utils.SharpenConv(3, stride=2)
        self.batch1 = nn.BatchNorm2d(11)  # params: 11 + 11 = 22

        self.conv2 = utils.DepthwiseSeparableConv(11, 22, (3, 3), stride=2) #7x7, params: [(3+3)*11] + [11*22*1*1] = 308
        self.batch2 = nn.BatchNorm2d(22) #params: 22 + 22 = 44

        self.conv3 = utils.DepthwiseSeparableConv(22, 44, (3, 3), stride=2) #3x3, params: [(3+3)*22] + [22*44*1*1] = 1100
        self.batch3 = nn.BatchNorm2d(44) #params: 44 + 44 = 88

        self.pool = nn.AvgPool2d((3,3))
        self.last = utils.LowRankClassifier(44, 100, 22) #params [44 * 22] + [22 * 100] = 3168


    def forward(self, x):
        input = x
        sobel_x = self.sobel_x(input)
        sobel_y = self.sobel_y(input)
        high_pass = self.high_pass(input)
        laplacian = self.laplacian(input)
        sharpen = self.sharpen(input)

        x = self.conv1(input)
        added_x = torch.cat((sobel_x, sobel_y, high_pass, laplacian, sharpen, x), 1)

        x = self.batch1(added_x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)

        x = self.pool(x)
        x = self.last(x)
        return x
