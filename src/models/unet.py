import torch
import torch.nn as nn
from typing import Any, List, Union
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
        super().__init__()
        num_down_channels = deepcopy(n_filters)
        num_down_channels.insert(0, in_channels)
        num_up_channels = deepcopy(n_filters)
        num_up_channels.insert(0, out_channels)
        self.num_scales = len(n_filters)
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

        return out