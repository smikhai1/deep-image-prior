from torch import nn
from typing import Union, Tuple


class _BaseBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x_in):
        x_out = self.block(x_in)
        return x_out


class DownsamplingBlock(_BaseBlock):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple]):
        """
        Downsampling block with configurable number of input/output channels and kernel sizes

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of the kernel
        """
        super().__init__()
        self.block = nn.Sequential(nn.ReflectionPad2d(padding=1),
                                   nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=0),
                                   nn.BatchNorm2d(out_channels),
                                   nn.LeakyReLU(inplace=True),
                                   nn.ReflectionPad2d(padding=1),
                                   nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=0),
                                   nn.BatchNorm2d(out_channels),
                                   nn.LeakyReLU(inplace=True)
                                   )


class SkipConnectionBlock(_BaseBlock):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple]):
        """
        Skip-connection block

        :param in_channels:
        :param kernel_size:
        """
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0),
                                   nn.BatchNorm2d(out_channels),
                                   nn.LeakyReLU(inplace=True)
                                   )

class UpsamplingBlock(_BaseBlock):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple],
                 upsampling: str = 'nearest'):
        """
        Upsampling block
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param upsampling: type of interpolation used in upsampling operation
        """
        super().__init__()
        Upsampler = nn.UpsamplingNearest2d if upsampling == 'nearest' else nn.UpsamplingBilinear2d
        self.block = nn.Sequential(nn.BatchNorm2d(in_channels),
                                   nn.ReflectionPad2d(padding=1),
                                   nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0),
                                   nn.BatchNorm2d(out_channels),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0),
                                   nn.BatchNorm2d(out_channels),
                                   nn.LeakyReLU(inplace=True),
                                   Upsampler(scale_factor=2)
                                   )