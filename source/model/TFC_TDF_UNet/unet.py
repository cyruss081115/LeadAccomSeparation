from typing import Tuple

import torch
import torch.nn as nn

from .unetBlocks import TFC_TDF_v1, TFC_TDF_v1_DownSample, TFC_TDF_v1_UpSample

class TFC_TDF_UNet_v1(nn.Module):
    r"""
    TFC-TDF intermediate block as described in:
    `Investigating U-Nets with various Intermediate Blocks for Spectrogram-based Singing Voice Separation`,
    arxiv: https://arxiv.org/abs/1912.02591
    
    
    Args:
        num_channels (int): number of input channels
        unet_depth (int): depth of the unet
        tfc_tdf_interal_layers (int): number of layers in each tfc_tdf block
        growth_rate (int): number of output channels of each tfc_tdf block
        kernel_size (Tuple[int, int]): kernel size of each tfc_tdf block
        frequency_bins (int): number of frequency bins
        bottleneck (int): number of bottleneck channels
        activation (str): activation function
        bias (bool): whether to use bias in convolutions
    """
    def __init__(self,
                 num_channels: int = 4,
                 unet_depth: int = 3,
                 tfc_tdf_interal_layers: int = 2,
                 growth_rate: int = 24,
                 kernel_size: Tuple[int, int] = (3, 3),
                 frequency_bins: int = 1024, # n_fft = 2048
                 bottleneck: int = 64, # freq bins // 16
                 activation: str = "ReLU",
                 bias: bool = False
        ):
        super(TFC_TDF_UNet_v1, self).__init__()
        self._check_init(num_channels, unet_depth, tfc_tdf_interal_layers, growth_rate, kernel_size, frequency_bins, bottleneck, activation, bias)

        self.in_conv = nn.Conv2d(num_channels, growth_rate, kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.relu = nn.ReLU()

        block_freq_bin_dimensions = [frequency_bins // (2 ** i) for i in range(unet_depth+1)]
        block_bottleneck_dimensions = [bottleneck // (2 ** i) for i in range(unet_depth+1)]

        self.down_blocks = nn.ModuleList()
        for blk_freq_bin_dim, blk_bn_dim in zip(
            block_freq_bin_dimensions[:unet_depth], block_bottleneck_dimensions[:unet_depth]
        ):
            self.down_blocks.append(
                TFC_TDF_v1_DownSample(
                    in_channels=growth_rate,
                    num_layers=tfc_tdf_interal_layers,
                    growth_rate=growth_rate,
                    kernel_size=kernel_size,
                    frequency_bins=blk_freq_bin_dim,
                    bottleneck=blk_bn_dim,
                    activation=activation,
                    bias=bias
                )
            )

        self.mid_block = TFC_TDF_v1(
            in_channels=growth_rate,
            num_layers=tfc_tdf_interal_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=block_freq_bin_dimensions[-1],
            bottleneck=block_bottleneck_dimensions[-1],
            activation=activation,
            bias=bias
        )

        self.up_blocks = nn.ModuleList()
        for blk_freq_bin_dim, blk_bn_dim in reversed(
            list(zip(block_freq_bin_dimensions[1:], block_bottleneck_dimensions[1:]))
        ):
            self.up_blocks.append(
                TFC_TDF_v1_UpSample(
                    in_channels=2*growth_rate,
                    num_layers=tfc_tdf_interal_layers,
                    growth_rate=growth_rate,
                    kernel_size=kernel_size,
                    frequency_bins=blk_freq_bin_dim,
                    bottleneck=blk_bn_dim,
                    activation=activation,
                    bias=bias
                )
            )
      
        self.out_conv = nn.Conv2d(growth_rate, num_channels, kernel_size=(1, 3), padding=(0, 1), stride=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass of the TFC-TDF UNet

        Args:
            x (torch.Tensor): Tensor of dimension (batch, channel, time, freq)

        Returns:
            torch.Tensor: Tensor of  dimension (batch, channel, time, freq)
        """
        x = self.in_conv(x)
        x = self.relu(x)

        skip_connections = []
        for down_block in self.down_blocks:
            x = down_block(x)
            skip_connections.append(x)

        x = self.mid_block(x)

        for up_block, skip_connection in zip(self.up_blocks, reversed(skip_connections)):
            cropped_skip_connection= skip_connection[:, :, :x.shape[2], :]
            x = torch.cat([x, cropped_skip_connection], dim=1)
            x = up_block(x)

        x = self.out_conv(x)
        return x

    def _check_init(self, num_channels, unet_depth, tfc_tdf_interal_layers, growth_rate, kernel_size, frequency_bins, bottleneck, activation, bias):
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if unet_depth <= 0:
            raise ValueError("unet_depth must be positive")
        if tfc_tdf_interal_layers <= 0:
            raise ValueError("tfc_tdf_interal_layers must be positive")
        if growth_rate <= 0:
            raise ValueError("growth_rate must be positive")
        if frequency_bins <= 0:
            raise ValueError("frequency_bins must be positive")
        if bottleneck <= 0:
            raise ValueError("bottleneck must be positive")

        if kernel_size[0] <= 0:
            raise ValueError("kernel_size[0] must be positive")
        if kernel_size[1] <= 0:
            raise ValueError("kernel_size[1] must be positive")
        
        if bottleneck % (2 ** unet_depth) != 0:
            raise ValueError("bottleneck must be divisible by 2 ** unet_depth")
        if frequency_bins % (2 ** unet_depth) != 0:
            raise ValueError("frequency_bins must be divisible by 2 ** unet_depth")


__all__ = ['TFC_TDF_UNet_v1']