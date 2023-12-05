import torch
import torch.nn as nn

from abc import ABC


class UNetBlock(nn.Module, ABC):
    def __init__(self):
        super(UNetBlock, self).__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

