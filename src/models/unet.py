import torch
import torch.nn as nn

from copy import deepcopy

from src.models import get_downsample_layer, get_skip_layer, get_upsample_layer

class UNet(nn.Module):

    def __init__(self, input_ch, out_ch, n_d, n_u, n_s, k_d, k_u, k_s, interpolation):
        """
        UNet-based model for denoising with skip-connections

        :param input_ch: int, number of channels in the fixed input
        :param input_ch: int, number of channels in the output
        :param n_d: list, numbers of filters in the downsampling layer
        :param n_u: list, numbers of filters in the upsampling layer
        :param n_s: list, numbers of filters in the skip-connection layer
        :param k_d: list, kernel sizes for the downsampling layer
        :param k_u: list, kernel sizes for the upsampling layer
        :param k_s: list, kernel sizes for the skip-connection layer
        :param interpolation: str, upsampling mode
        """
        super().__init__()

        self.encoder = []
        self.encoder.append(get_downsample_layer(in_channels=input_ch,
                                                 num_filters=n_d[0],
                                                 kernel_size=k_d[0]
                                                 )
                            )
        for i in range(1, len(n_d)):
            self.encoder.append(get_downsample_layer(in_channels=n_d[i-1],
                                                     num_filters=n_d[i],
                                                     kernel_size=k_d[i]
                                                     )
                                )

        self.decoder = []
        self.decoder.append(get_upsample_layer(in_channels=n_s[-1],
                                               num_filters=n_u[0],
                                               kernel_size=k_u[0],
                                               interpolation=interpolation
                                               )
                            )
        for i in range(1, len(n_u)):
            self.decoder.append(get_upsample_layer(in_channels=n_u[i-1]+n_s[-i],
                                                   num_filters=n_u[i],
                                                   kernel_size=k_u[i],
                                                   interpolation=interpolation
                                                  )
                                )
        end = nn.Sequential(nn.Conv2d(in_channels=n_u[-1], out_channels=out_ch, kernel_size=1),
                            nn.Sigmoid()
                            )
        self.decoder.append(end)

        self.skipper = [get_skip_layer(in_channels=n_d[i], num_filters=n_s[i], kernel_size=k_s[i]) if n_s[i] is not None else None
                        for i in range(len(n_s) )
                        ]

    def forward(self, x):

        out = deepcopy(x)

        encoder_outs = []
        for i in range(len(self.encoder)):
            out = self.encoder[i].forward(out)
            encoder_outs.append(out)

        #out = self.decoder[0].forward(out)

        out = self.skipper[-1].forward(out)
        out = self.decoder[0].forward(out)
        for i in range(1, len(self.decoder)-1):
            if self.skipper[-i-1] is not None:
                skip = self.skipper[-i-1].forward(encoder_outs[-i-1])
                out = self.decoder[i].forward(torch.cat((out, skip), dim=1))
            else:
                out = self.decoder[i].forward(out)
        out = self.decoder[-1].forward(out)

        return out

    def parameters(self):

        params = []
        for block in self.encoder:
            params.extend(list(block.parameters()))

        for block in self.skipper:
            params.extend(list(block.parameters()))

        for block in self.decoder:
            params.extend(list(block.parameters()))

        return params

    def to(self, device):

        for block in self.encoder:
            block.to(device)

        for block in self.skipper:
            block.to(device)

        for block in self.decoder:
            block.to(device)