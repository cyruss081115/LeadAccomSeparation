from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..unet_blocks import UNetBlock
from ..basic_blocks import (
    BasicBlock,
    TimeFrequencyConvolutionBlock,
    TimeDistributedSelfAttentionBlock,
    DownSample2DBlock,
    UpSample2DBlock
)

class TFC_TDSA(UNetBlock):
    r"""
    Time Frequency Convolution + Time Distributed Self Attention, using a similar architecture as TFC-TDF described in:
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
                 num_attention_heads: int,
                 use_vanilla_self_attention: bool = False,
                 activation: str = "ReLU",
                 bias: bool = False
        ):
        super(TFC_TDSA, self).__init__()
        self.tfc_block = TimeFrequencyConvolutionBlock(
            in_channels=in_channels,
            growth_rate=growth_rate,
            num_layers=num_layers,
            kernel_size=kernel_size,
            activation=activation,
            bias=bias
        )
        self.tdsa_block = TimeDistributedSelfAttentionBlock(
            embed_dim=frequency_bins,
            num_heads=num_attention_heads,
            num_layers=num_layers,
            activation=activation,
            use_vanilla_self_attention=use_vanilla_self_attention,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tfc_block(x)
        return x + self.tdsa_block(x)


class TFC_TDSA_DownSample(UNetBlock):
    r"""
    Downsample block using Time Frequency Convolution + Time Distributed Self Attention, using a similar architecture as TFC-TDF described in:
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
                 num_attention_heads: int,
                 use_vanilla_self_attention: bool = False,
                 activation: str = "ReLU",
                 bias: bool = False
        ):
        super(TFC_TDSA_DownSample, self).__init__()
        self.tfc_tdsa_block = TFC_TDSA(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            num_attention_heads=num_attention_heads,
            use_vanilla_self_attention=use_vanilla_self_attention,
            activation=activation,
            bias=bias
        )
        self.downsample_block = DownSample2DBlock(
            in_channels=growth_rate,
            out_channels=growth_rate,
            activation=activation,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tfc_tdsa_block(x)
        return self.downsample_block(x)


class TFC_TDSA_UpSample(UNetBlock):
    def __init__(self,
                 in_channels: int,
                 num_layers: int,
                 growth_rate: int,
                 kernel_size: Tuple[int, int],
                 frequency_bins: int,
                 num_attention_heads: int,
                 use_vanilla_self_attention: bool = False,
                 activation: str = "ReLU",
                 bias: bool = False
        ):
        super(TFC_TDSA_UpSample, self).__init__()
        self.tfc_tdsa_block = TFC_TDSA(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            num_attention_heads=num_attention_heads,
            use_vanilla_self_attention=use_vanilla_self_attention,
            activation=activation,
            bias=bias
        )
        self.upsample_block = UpSample2DBlock(
            in_channels=growth_rate,
            out_channels=growth_rate,
            activation=activation,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tfc_tdsa_block(x)
        return self.upsample_block(x)