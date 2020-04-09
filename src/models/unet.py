import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
from copy import deepcopy

from src.models.layers import DownsamplingBlock, SkipConnectionBlock, UpsamplingBlock

class UNet(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_filters: List[int],
                 k_d: Union[List[int], int],
                 k_u: Union[List[int], int],
                 n_skips: List[int],
                 k_s: Union[List[int], int],
                 upsampling: str = 'nearest'
                 ):
        """
        Hourglass-like model for denoising in Deep Image Prior framework.

        :param in_channels: int, number of channels in the input tensor
        :param out_channels: int, number of channels in the output tensor
        :param n_filters: list, list that contains the number of filters used in each block
        :param k_d: int or list, sets the kernel size in each downsampling block; if int is given,
                    the kernel size will be the same in each block
        :param k_u: int or list, sets the kernel size in each upsampling block; if int is given,
                    the kernel size will be the same in each block
        :param n_skips: list, sets the number of channels in the skip-connection block; 0 corresponds
                        to the absence of skip-connection
        :param k_s: list or int, sets the kernel size in each skip-connection block; if int is given,
                    the kernel size will be the same in each block
        :param upsampling: str, sets the upsampling mode ['nearest', 'bicubic', 'bilinear']
        """
        super().__init__()
        self.num_scales = len(n_filters)

        if isinstance(k_d, int):
            k_d = [k_d] * self.num_scales
        if isinstance(k_u, int):
            k_u = [k_u] * self.num_scales
        if isinstance(k_s, int):
            k_s = [k_s] * self.num_scales

        if not self.num_scales == len(k_d) == len(k_u) == len(k_s) == len(n_skips):
            raise ValueError('There are incorrect arguments in the constructor!')

        num_down_channels = deepcopy(n_filters)
        num_down_channels.insert(0, in_channels)
        num_up_channels = deepcopy(n_filters)
        num_up_channels.insert(0, out_channels)

        for i in range(self.num_scales):
            setattr(self,
                    'down_block_{}'.format(i),
                    DownsamplingBlock(num_down_channels[i], num_down_channels[i+1], k_d[i])
                    )
            if n_skips[i] != 0:
                setattr(self,
                        'skip_block_{}'.format(i),
                        SkipConnectionBlock(num_down_channels[i+1], n_skips[i], k_s[i])
                        )
                num_up_in_channels = num_up_channels[i+1] + n_skips[i]
            else:
                num_up_in_channels = num_up_channels[i+1]
            setattr(self,
                    'up_block_{}'.format(i),
                    UpsamplingBlock(num_up_in_channels, num_up_channels[i], k_u[i], upsampling=upsampling)
                    )

    def forward(self, x):

        out = x
        skips_outs = []

        # downsampling and skip-connections paths
        for i in range(self.num_scales):
            out = getattr(self, 'down_block_{}'.format(i))(out)
            try:
                skip_out = getattr(self, 'skip_block_{}'.format(i))(out)
                skips_outs.append(skip_out)
            except AttributeError:
                skips_outs.append(None)

        # upsampling path
        for i in reversed(range(self.num_scales)):
            if skips_outs[i] is not None:
                out = torch.cat((out, skips_outs[i]), dim=1)
            out = getattr(self, 'up_block_{}'.format(i))(out)

        out = F.tanh(out)

        return out
