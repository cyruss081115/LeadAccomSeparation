import torch
import torch.nn as nn

from abc import ABC

from .basic_blocks import DownSample2DBlock, UpSample2DBlock


class UNetBlock(nn.Module, ABC):
    def __init__(self):
        super(UNetBlock, self).__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class UNetDownSampleBlock(UNetBlock):
    def __init__(self, unet_block: UNetBlock, down_block: DownSample2DBlock):
        super(UNetDownSampleBlock, self).__init__()
        self.unet_block = unet_block
        self.down_block = down_block
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unet_block(x)
        return self.down_block(x)

class UNetUpSampleBlock(UNetBlock):
    def __init__(self, unet_block: UNetBlock, up_block: UpSample2DBlock):
        super(UNetUpSampleBlock, self).__init__()
        self.unet_block = unet_block
        self.up_block = up_block
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unet_block(x)
        return self.up_block(x)