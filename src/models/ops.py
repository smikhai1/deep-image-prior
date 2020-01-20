import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 pad,
                 padding_type="constant",
                 use_bias=True
                 ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.padding_type = padding_type

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

    def forward(self, x):

        if self.pad > 0:
            out = F.pad(x, pad=[self.pad]*4, mode=self.padding_type)
        out = F.conv2d(out, weight=self.weight, bias=self.bias, stride=self.stride)

        return out