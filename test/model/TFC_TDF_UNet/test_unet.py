import sys, os
sys.path.append(os.getcwd())

import unittest

import torch
import torch.nn as nn

from source.model.TFC_TDF_UNet.unet import TFC_TDF_UNet_v1
from source.model.TFC_TDF_UNet.unetBlocks import (
    TFC_TDF_v1,
    TFC_TDF_v1_DownSample,
    TFC_TDF_v1_UpSample,
)

class TestTFC_TDF_UNet_v1(unittest.TestCase):
    def test_init_is_nn_Module(self):
        num_channels = 2
        unet_depth = 3
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        bottleneck = 64
        activation = "ReLU"
        bias = False

        unet = TFC_TDF_UNet_v1(
            num_channels=num_channels,
            unet_depth=unet_depth,
            tfc_tdf_interal_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            bottleneck=bottleneck,
            activation=activation,
            bias=bias
        )

        self.assertIsInstance(unet, nn.Module)
    
    def test_init_architecture(self):
        num_channels = 2
        unet_depth = 3
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        bottleneck = 64
        activation = "LeakyReLU"
        bias = False

        unet = TFC_TDF_UNet_v1(
            num_channels=num_channels,
            unet_depth=unet_depth,
            tfc_tdf_interal_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            bottleneck=bottleneck,
            activation=activation,
            bias=bias
        )
        
        self.assertIsInstance(unet.in_conv, nn.Conv2d)
        self.assertEqual(unet.in_conv.in_channels, num_channels)
        self.assertEqual(unet.in_conv.out_channels, growth_rate)
        self.assertEqual(unet.in_conv.kernel_size, (1, 3))
        self.assertEqual(unet.in_conv.padding, (0, 1))
        
        self.assertIsInstance(unet.relu, nn.ReLU)
        
        # Test DownSample Blocks
        self.assertIsInstance(unet.down_blocks, nn.ModuleList)
        self.assertEqual(len(unet.down_blocks), unet_depth)
        for i, down_block in enumerate(unet.down_blocks):
            self.assertIsInstance(down_block, TFC_TDF_v1_DownSample)
            expected_downsample_block = TFC_TDF_v1_DownSample(
                in_channels=growth_rate,
                num_layers=num_layers,
                growth_rate=growth_rate,
                kernel_size=kernel_size,
                frequency_bins=frequency_bins // (2 ** i),
                bottleneck=bottleneck // (2 ** i),
                activation=activation,
                bias=bias
            )
            self.assertEqual(down_block.__str__(), expected_downsample_block.__str__())
        
        # Test Mid Block
        self.assertIsInstance(unet.mid_block, TFC_TDF_v1)
        expected_mid_block = TFC_TDF_v1(
            in_channels=growth_rate,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins // (2 ** unet_depth),
            bottleneck=bottleneck // (2 ** unet_depth),
            activation=activation,
            bias=bias
        )
        self.assertEqual(unet.mid_block.__str__(), expected_mid_block.__str__())
        
        # Test UpSample Blocks
        self.assertIsInstance(unet.up_blocks, nn.ModuleList)
        self.assertEqual(len(unet.up_blocks), unet_depth)
        for i, up_block in enumerate(unet.up_blocks):
            self.assertIsInstance(up_block, TFC_TDF_v1_UpSample)
            expected_upsample_block = TFC_TDF_v1_UpSample(
                in_channels=2*growth_rate,
                num_layers=num_layers,
                growth_rate=growth_rate,
                kernel_size=kernel_size,
                frequency_bins=(frequency_bins // (2 ** (unet_depth - i))),
                bottleneck=(bottleneck // (2 ** (unet_depth - i))),
                activation=activation,
                bias=bias
            )
            self.assertEqual(up_block.__str__(), expected_upsample_block.__str__())
    
    def test_forward_output_dims(self):
        num_channels = 2
        unet_depth = 3
        num_layers = 2
        growth_rate = 24
        kernel_size = (3, 3)
        frequency_bins = 1024
        bottleneck = 64
        activation = "LeakyReLU"
        bias = False

        device = (
            "cuda" if torch.cuda.is_available() else 
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        unet = TFC_TDF_UNet_v1(
            num_channels=num_channels,
            unet_depth=unet_depth,
            tfc_tdf_interal_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            bottleneck=bottleneck,
            activation=activation,
            bias=bias
        ).to(device)

        x = torch.rand([1, 2, 352, 1024]).to(device) #
        y = unet(x)

        self.assertEqual(list(y.shape), [1, 2, 352, 1024])


if __name__ == "__main__":
    unittest.main()