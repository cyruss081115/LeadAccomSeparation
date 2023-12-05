from abc import ABC
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..unetBlock import UnetBlock
from ..basicBlocks import (
    BasicBlock,
    TimeFrequencyConvolutionBlock,
    TimeDistributedFullyConnectedBlock,
    DownSample2DBlock,
    UpSample2DBlock
)



class TFC_TDF_v1(UnetBlock):
    r"""
    TFC-TDF intermediate block as described in:
    `Investigating U-Nets with various Intermediate Blocks for Spectrogram-based Singing Voice Separation`,
    arxiv: https://arxiv.org/abs/1912.02591

    Args:
        in_channels (int): Number of input channels
        num_layers (int): Number of layers in each block
        growth_rate (int): Number of channels to add per layer, also the output of this block
        kernel_size (Tuple[int, int]): Size of the convolutional kernel
        frequency_bins (int): Number of frequency bins in the input
        bottleneck (int): Number of channels in the bottleneck layer
        activation (str, optional): Activation function to use. Defaults to "ReLU".
        bias (bool, optional): Whether to use bias in the convolutional layers. Defaults to False.
    """
    def __init__(self,
                 in_channels: int,
                 num_layers: int,
                 growth_rate: int,
                 kernel_size: Tuple[int, int],
                 frequency_bins: int,
                 bottleneck: int,
                 activation: str = "ReLU",
                 bias: bool = False
        ):
        super(TFC_TDF_v1, self).__init__()
        self.tfc_block = TimeFrequencyConvolutionBlock(
            in_channels=in_channels,
            growth_rate=growth_rate,
            num_layers=num_layers,
            kernel_size=kernel_size,
            activation=activation,
            bias=bias
        )
        self.tdf_block = TimeDistributedFullyConnectedBlock(
            channels=growth_rate,
            frequency_bins=frequency_bins,
            bottleneck=bottleneck,
            num_layers=num_layers,
            activation=activation,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tfc_block(x)
        return x + self.tdf_block(x)


class TFC_TDF_v1_DownSample(UnetBlock):
    r"""
    TFC-TDF intermediate block as described in:
    `Investigating U-Nets with various Intermediate Blocks for Spectrogram-based Singing Voice Separation`,
    arxiv: https://arxiv.org/abs/1912.02591
    """
    def __init__(self,
                 in_channels: int,
                 num_layers: int,
                 growth_rate: int,
                 kernel_size: Tuple[int, int],
                 frequency_bins: int,
                 bottleneck: int,
                 activation: str = "ReLU",
                 bias: bool = False
        ):
        super(TFC_TDF_v1_DownSample, self).__init__()
        self.tfc_tdf_block = TFC_TDF_v1(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            bottleneck=bottleneck,
            activation=activation,
            bias=bias
        )
        self.downsample = DownSample2DBlock(
            in_channels=growth_rate,
            out_channels=growth_rate,
            activation=activation,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tfc_tdf_block(x)
        return self.downsample(x)


class TFC_TDF_v1_UpSample(UnetBlock):
    r"""
    TFC-TDF intermediate block as described in:
    `Investigating U-Nets with various Intermediate Blocks for Spectrogram-based Singing Voice Separation`,
    arxiv: https://arxiv.org/abs/1912.02591
    """
    def __init__(self,
                 in_channels: int,
                 num_layers: int,
                 growth_rate: int,
                 kernel_size: Tuple[int, int],
                 frequency_bins: int,
                 bottleneck: int,
                 activation: str = "ReLU",
                 bias: bool = False
        ):
        super(TFC_TDF_v1_UpSample, self).__init__()
        self.tfc_tdf_block = TFC_TDF_v1(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            bottleneck=bottleneck,
            activation=activation,
            bias=bias
        )
        self.upsample = UpSample2DBlock(
            in_channels=growth_rate,
            out_channels=growth_rate,
            activation=activation,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tfc_tdf_block(x)
        return self.upsample(x)