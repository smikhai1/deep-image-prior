import torch
import torch.nn as nn

from src.models import Conv2d

def get_downsample_layer(in_channels, num_filters, kernel_size):

    layers = nn.Sequential(nn.ReflectionPad2d(kernel_size//2),
                           nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size, stride=2, bias=False),
                           nn.BatchNorm2d(num_filters),
                           nn.LeakyReLU(),
                           nn.ReflectionPad2d(kernel_size//2),
                           nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size,
                                     stride=1, bias=False),
                           nn.BatchNorm2d(num_filters),
                           nn.LeakyReLU()
                           )

    return layers

def get_upsample_layer(in_channels, num_filters, kernel_size, interpolation):

    upsample = nn.Upsample(scale_factor=2, mode=interpolation)

    layers = nn.Sequential(nn.BatchNorm2d(in_channels),
                           nn.ReflectionPad2d(kernel_size // 2),
                           nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size,
                                     stride=1, bias=False),
                           nn.BatchNorm2d(num_filters),
                           nn.LeakyReLU(),
                           nn.ReflectionPad2d(kernel_size // 2),
                           nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size,
                                     stride=1, bias=False),
                           nn.BatchNorm2d(num_filters),
                           nn.LeakyReLU(),
                           upsample
                           )


    return layers

def get_skip_layer(in_channels, num_filters, kernel_size):

    layers = nn.Sequential(nn.ReflectionPad2d(kernel_size // 2),
                           nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size,
                                     stride=1, bias=False),
                           nn.BatchNorm2d(num_filters),
                           nn.LeakyReLU()
                           )

    return layers