from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..unet_blocks import UNetBlock, UNetDownSampleBlock, UNetUpSampleBlock
from ..basic_blocks import (
    BasicBlock,
    TimeFrequencyConvolutionBlock,
    TimeDistributedTransformerBlock,
    DownSample2DBlock,
    UpSample2DBlock
)

class TFC_TDT(UNetBlock):
    r"""
    Time Frequency Convolution + Time Distributed Transformer, using a similar architecture as TFC-TDF described in:
    `Investigating U-Nets with various Intermediate Blocks for Spectrogram-based Singing Voice Separation`,
    arxiv: https://arxiv.org/abs/1912.02591

    Args:
        in_channels (int): Number of input channels
        num_layers (int): Number of layers in each block
        growth_rate (int): Number of channels to add per layer, also the output of this block
        kernel_size (Tuple[int, int]): Size of the convolutional kernel
        num_attention_heads (int): Number of attention heads
        activation (str, optional): Activation function to use. Defaults to "ReLU".
        bias (bool, optional): Whether to use bias. Defaults to False.
    """

    def __init__(self,
                 in_channels: int,
                 num_layers: int,
                 growth_rate: int,
                 kernel_size: Tuple[int, int],
                 frequency_bins: int,
                 dropout: float = 0.2,
                 activation: str = "ReLU",
                 bias: bool = False
        ):
        super(TFC_TDT, self).__init__()
        self.tfc_block = TimeFrequencyConvolutionBlock(
            in_channels=in_channels,
            growth_rate=growth_rate,
            num_layers=num_layers,
            kernel_size=kernel_size,
            activation=activation,
            bias=bias
        )
        self.tdt_block = TimeDistributedTransformerBlock(
            embed_dim=growth_rate,
            num_layers=num_layers,
            dropout=dropout,
            seq_length=frequency_bins,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tfc_block(x)
        return x + self.tdt_block(x)


class TFC_TDT_DownSample(UNetDownSampleBlock):
    def __init__(self,
                 in_channels: int,
                 num_layers: int,
                 growth_rate: int,
                 kernel_size: Tuple[int, int],
                 frequency_bins: int,
                 dropout: float = 0.2,
                 activation: str = "ReLU",
                 bias: bool = False
        ):
        unet_block = TFC_TDT(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            dropout=dropout,
            activation=activation,
            bias=bias
        )
        down_block = DownSample2DBlock(
            in_channels=growth_rate,
            out_channels=growth_rate,
            activation=activation,
            bias=bias
        )
        super(TFC_TDT_DownSample, self).__init__(unet_block=unet_block, down_block=down_block)


class TFC_TDT_UpSample(UNetUpSampleBlock):
    def __init__(self,
                 in_channels: int,
                 num_layers: int,
                 growth_rate: int,
                 kernel_size: Tuple[int, int],
                 frequency_bins: int,
                 dropout: float = 0.2,
                 activation: str = "ReLU",
                 bias: bool = False
        ):
        unet_block = TFC_TDT(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            dropout=dropout,
            activation=activation,
            bias=bias
        )
        up_block = UpSample2DBlock(
            in_channels=growth_rate,
            out_channels=growth_rate,
            activation=activation,
            bias=bias
        )
        super(TFC_TDT_UpSample, self).__init__(unet_block=unet_block, up_block=up_block)