import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=False):
        super().__init__()
        self.depthwise = SparseConv(in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, stride=stride)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SparseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=None, stride=1, padding=0):
        super().__init__()
        if groups is None:
            groups = in_channels  # default to depthwise
        assert in_channels == groups, "SparseConv is designed for depthwise conv (1 filter per channel)"

        self.kH = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.kW = kernel_size if isinstance(kernel_size, int) else kernel_size[1]

        self.col_kernel = nn.Parameter(torch.randn(in_channels, 1, self.kH, 1))
        self.row_kernel = nn.Parameter(torch.randn(in_channels, 1, 1, self.kW))

        self.padding = padding
        self.stride = stride
        self.groups = groups

    def forward(self, x):
        # (C, 1, kH, 1) * (C, 1, 1, kW) → (C, 1, kH, kW)
        weight = self.col_kernel * self.row_kernel
        return F.conv2d(x, weight, bias=None, stride=self.stride, padding=self.padding, groups=self.groups)

class LowRankClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.reduce = nn.Conv2d(in_dim, rank, kernel_size=1, bias=False)  # (in r)
        self.expand = nn.Conv2d(rank, out_dim, kernel_size=1, bias=False)  # (r 100)

    def forward(self, x):  # x: (B, in_dim, 1, 1)
        x = self.reduce(x)  #  (B, r, 1, 1)
        x = self.expand(x)  #  (B, 100, 1, 1)
        return x.view(x.size(0), -1)  # (B, 100)

class SobelXConv(nn.Module):
    def __init__(self, in_channels, stride=2, padding=0):
        super().__init__()
        sobel_x = torch.empty((1, in_channels, 3, 3))
        sobel_x[0,:,2] = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.register_buffer("sobel_kernel", sobel_x)
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        return F.conv2d(x, self.sobel_kernel, bias=None, stride=self.stride, padding=self.padding)

class SobelYConv(nn.Module):
    def __init__(self, in_channels, stride=2, padding=0):
        super().__init__()
        sobel_y = torch.empty((1, in_channels, 3, 3))
        sobel_y[0, :, 2] = torch.tensor([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
        self.register_buffer("sobel_kernel", sobel_y)
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        return F.conv2d(x, self.sobel_kernel, bias=None, stride=self.stride, padding=self.padding)

class HighPassConv(nn.Module):
    def __init__(self, in_channels, stride=2, padding=0):
        super().__init__()
        high_pass = torch.empty((1, in_channels, 3, 3))
        high_pass[0, :, 2] = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        self.register_buffer("high_pass_kernel", high_pass)
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        return F.conv2d(x, self.high_pass_kernel, bias=None, stride=self.stride, padding=self.padding)

class SharpenConv(nn.Module):
    def __init__(self, in_channels, stride=2, padding=0):
        super().__init__()
        sharpen = torch.empty((1, in_channels, 3, 3))
        sharpen[0, :, 2] = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.register_buffer("sharpen_kernel", sharpen)
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        return F.conv2d(x, self.sharpen_kernel, bias=None, stride=self.stride, padding=self.padding)

class LaplacianConv(nn.Module):
    def __init__(self, in_channels, stride=2, padding=0):
        super().__init__()
        laplacian = torch.empty((1, in_channels, 3, 3))
        laplacian[0, :, 2] = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        self.register_buffer("laplacian_kernel", laplacian)
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        return F.conv2d(x, self.laplacian_kernel, bias=None, stride=self.stride, padding=self.padding)