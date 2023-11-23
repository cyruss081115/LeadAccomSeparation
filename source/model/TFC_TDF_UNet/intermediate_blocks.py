from typing import Literal
from abc import ABC

import torch
import einops
import torch.nn as nn
import torch.nn.functional as F


def get_activation(
        activation: Literal['ReLU', 'LeakyReLU', 'SiLU']) -> nn.Module:
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU()
    elif activation == 'SiLU':
        return nn.SiLU()
    else:
        raise Exception


class IntermediateBlock(nn.Module, ABC):
    r"""
    Abstract class for intermediate blocks.
    """
    def __init__(self):
        super(IntermediateBlock, self).__init__()
        pass

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        r"""
        Forward propagate the feature tensor.
        Args:
            feature (torch.Tensor): Tensor of feature of dimension (b, c, n_fft, n_frames)
        Returns:
            torch.Tensor: Tensor of feature of dimension (b, c, n_fft, n_frames)
        """
        pass


class TDFBlock(IntermediateBlock):
    r"""
    Time-distributed fully connected block. It will be applied to each channel of each frame separately and identically.
    Different from the original paper, batch normalization is applied before the fully connected layer.
    [batch, channels, n_frames, frequency_bins] -> [batch, channels, n_frames, frequency_bins]

    Args:
        channels (int): Number of channels.
        frequency_bins (int): Number of frequency bins, same as n_fft in STFT.
        bottleneck (int): Bottleneck dimension, it should be a divisor of frequency_bins.
        num_layers (int): Number of layers.
        activation (str, optional): Activation function. Defaults to 'ReLU'.
        bias (bool, optional): Whether to use bias in the fully connected layer. Defaults to False.
    """
    def __init__(self,
                 channels: int,
                 frequency_bins: int,
                 bottleneck: int,
                 num_layers: int,
                 activation: str = 'ReLU',
                 bias: bool = False):
        super(TDFBlock, self).__init__()
        self._check_init(channels, frequency_bins, bottleneck, num_layers, activation, bias)
        self.norm_fc_act_stack = nn.ModuleList()

        if num_layers == 1:
            self.norm_fc_act_stack.append(
                nn.Sequential(
                    nn.BatchNorm2d(channels),
                    nn.Linear(frequency_bins, frequency_bins, bias=bias),
                    get_activation(activation),
                )
            )
        else:
            for i in range(num_layers):
                if i == 0:
                    self.norm_fc_act_stack.append(
                        nn.Sequential(
                            nn.BatchNorm2d(channels),
                            nn.Linear(frequency_bins, bottleneck, bias=bias),
                            get_activation(activation),
                        )
                    )
                elif i == num_layers - 1:
                    self.norm_fc_act_stack.append(
                        nn.Sequential(
                            nn.BatchNorm2d(channels),
                            nn.Linear(bottleneck, frequency_bins, bias=bias),
                            get_activation(activation),
                        )
                    )
                else:
                    self.norm_fc_act_stack.append(
                        nn.Sequential(
                            nn.BatchNorm2d(channels),
                            nn.Linear(bottleneck, bottleneck, bias=bias),
                            get_activation(activation),
                        )
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward propagate the feature tensor.
        Args:
            x (torch.Tensor): Tensor of feature of dimension (b, c, n_frames, n_fft)

        Returns:
            torch.Tensor: Tensor of feature of dimension (b, c, n_frames, n_fft)
        """
        identity = x
        for norm_fc_act in self.norm_fc_act_stack:
            x = norm_fc_act(x)
        return x + identity

    def _check_init(self, channels, frequency_bins, bottleneck, num_layers, activation, bias):
        if num_layers <= 0:
            raise ValueError("The number of layers must be greater than 0.")
        if bottleneck <= 0:
            raise ValueError("The bottleneck dimension must be greater than 0.")
        if frequency_bins % bottleneck != 0:
            raise ValueError("The number of frequency bins must be divisible by the bottleneck dimension.")


class TDCBlock(IntermediateBlock):
    r"""
    Time-distributed convolution block. It has a DenseNet-like architecture.
    [batch, channels, n_frames, frequency_bins] -> [batch, growth_rate, n_frames, frequency_bins]

    Args:
        in_channels (int): Number of input channels.
        growth_rate (int): Number of output channels.
        num_layers (int): Number of layers.
        kernel_size (int): Kernel size of the convolution layer. It should be odd to keep input freq dim and output freq dim identical.
        activation (str, optional): Activation function. Defaults to 'ReLU'.
        bias (bool, optional): Whether to use bias in the convolution layer. Defaults to False.
    """
    def __init__(self,
                 in_channels: int,
                 growth_rate: int,
                 num_layers: int,
                 kernel_size: int,
                 activation: str = 'ReLU',
                 bias: bool = False):
        super(TDCBlock, self).__init__()
        # self._check_init(in_channels, growth_rate, num_layers, kernel_size, activation, bias)
        self.norm_conv_act_stack = nn.ModuleList()

        current_channels = in_channels
        for i in range(num_layers):
            self.norm_conv_act_stack.append(
                nn.Sequential(
                    nn.BatchNorm1d(current_channels),
                    nn.Conv1d(current_channels, growth_rate, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
                    get_activation(activation),
                )
            )
            current_channels += growth_rate
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward propagate the feature tensor.
        [batch, channels, n_frames, frequency_bins] -> [batch, growth_rate, n_frames, frequency_bins]

        Args:
            x (torch.Tensor): Tensor of feature of dimension (b, c, n_frames, n_fft)
        
        Returns:
            torch.Tensor: Tensor of feature of dimension (b, growth_rate, n_frames, n_fft)
        """
        batch, channels, n_frames, frequency_bins = x.shape
        x = einops.rearrange(x, 'b c t f -> (b t) c f')
        x_cat = x
        for i, norm_conv_act in enumerate(self.norm_conv_act_stack):
            x = norm_conv_act(x_cat)
            if i != len(self.norm_conv_act_stack) - 1:
                x_cat = torch.cat([x_cat, x], dim=1)
        x = einops.rearrange(x, '(b t) c f -> b c t f', b=batch, t=n_frames)
        return x
    
    def _check_init(self, in_channels, growth_rate, num_layers, kernel_size, activation, bias):
        if num_layers <= 0:
            raise ValueError("The number of layers must be greater than 0.")
        if growth_rate <= 0:
            raise ValueError("The growth rate must be greater than 0.")
        if kernel_size <= 0:
            raise ValueError("The kernel size must be greater than 0.")
        if kernel_size % 2 != 1:
            raise ValueError("The kernel size must be odd to keep input freq dim and output freq dim identical.")