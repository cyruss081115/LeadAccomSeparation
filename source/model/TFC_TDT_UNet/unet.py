from typing import Tuple

import torch
import torch.nn as nn

from .building_blocks import TFC_TDT, TFC_TDT_DownSample, TFC_TDT_UpSample

class TFC_TDT_UNet(nn.Module):
    r"""
    Self Attention variant of TFC-TDT intermediate block as described in:
    `Investigating U-Nets with various Intermediate Blocks for Spectrogram-based Singing Voice Separation`,
    arxiv: https://arxiv.org/abs/1912.02591
    
    
    Args:
        
    """
    def __init__(self,
                 num_channels: int = 4,
                 unet_depth: int = 3,
                 tfc_tdt_internal_layers: int = 2,
                 growth_rate: int = 24,
                 kernel_size: Tuple[int, int] = (3, 3),
                 frequency_bins: int = 1024, # n_fft = 2048
                 dropout: float = 0.2,
                 activation: str = "ReLU",
                 bias: bool = False
        ):
        super(TFC_TDT_UNet, self).__init__()
        self._check_init(num_channels, unet_depth, tfc_tdt_internal_layers, growth_rate, kernel_size, frequency_bins, dropout, activation, bias)

        self.in_conv = nn.Conv2d(num_channels, growth_rate, kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.relu = nn.ReLU()

        block_freq_bin_dimensions = [frequency_bins // (2 ** i) for i in range(unet_depth+1)]

        self.down_blocks = nn.ModuleList()
        for blk_freq_bin_dim in block_freq_bin_dimensions[:unet_depth]:
            self.down_blocks.append(
                TFC_TDT_DownSample(
                    in_channels=growth_rate,
                    num_layers=tfc_tdt_internal_layers,
                    growth_rate=growth_rate,
                    kernel_size=kernel_size,
                    frequency_bins=blk_freq_bin_dim,
                    dropout=dropout,
                    activation=activation,
                    bias=bias
                )
            )

        self.mid_block = TFC_TDT(
            in_channels=growth_rate,
            num_layers=tfc_tdt_internal_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=block_freq_bin_dimensions[-1],
            dropout=dropout,
            activation=activation,
            bias=bias
        )

        self.up_blocks = nn.ModuleList()
        for blk_freq_bin_dim in reversed(block_freq_bin_dimensions[1:]):
            self.up_blocks.append(
                TFC_TDT_UpSample(
                    in_channels=2*growth_rate,
                    num_layers=tfc_tdt_internal_layers,
                    growth_rate=growth_rate,
                    kernel_size=kernel_size,
                    frequency_bins=blk_freq_bin_dim,
                    dropout=dropout,
                    activation=activation,
                    bias=bias
                )
            )
      
        self.out_conv = nn.Conv2d(growth_rate, num_channels, kernel_size=(1, 3), padding=(0, 1), stride=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass of the TFC-TDT UNet

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

    def _check_init(self, num_channels, unet_depth, tfc_tdsa_interal_layers, growth_rate, kernel_size, frequency_bins, num_attention_heads, activation, bias):
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if unet_depth <= 0:
            raise ValueError("unet_depth must be positive")
        if tfc_tdsa_interal_layers <= 0:
            raise ValueError("tfc_tdsa_interal_layers must be positive")
        if growth_rate <= 0:
            raise ValueError("growth_rate must be positive")
        if frequency_bins <= 0:
            raise ValueError("frequency_bins must be positive")
        if num_attention_heads <= 0:
            raise ValueError("Number of attention heads must be positive")

        if kernel_size[0] <= 0:
            raise ValueError("kernel_size[0] must be positive")
        if kernel_size[1] <= 0:
            raise ValueError("kernel_size[1] must be positive")

        if frequency_bins % (2 ** unet_depth) != 0:
            raise ValueError("frequency_bins must be divisible by 2 ** unet_depth")


__all__ = ['TFC_TDT_UNet']