import sys, os
sys.path.append(os.getcwd())

import unittest

import torch
import torch.nn as nn

from source.model.TFC_TDF_UNet.basicBlocks import (
    TimeFrequencyConvolutionBlock,
    TimeDistributedFullyConnectedBlock,
    DownSample2DBlock,
    UpSample2DBlock,
)
from source.model.TFC_TDF_UNet.unetBlocks import (
    TFC_TDF_v1,
    TFC_TDF_v1_DownSample,
    TFC_TDF_v1_UpSample,
)


class TestIntermediateBlock(unittest.TestCase):
    def test_init_is_nn_Module(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        bottleneck = 4
        activation = "ReLU"
        bias = False

        tfc_tdf_block = TFC_TDF_v1(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            bottleneck=bottleneck,
            activation=activation,
            bias=bias
        )

        self.assertIsInstance(tfc_tdf_block, nn.Module)
    
    def test_init_architecture(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        bottleneck = 4
        activation = "ReLU"
        bias = False

        tfc_tdf_block = TFC_TDF_v1(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            bottleneck=bottleneck,
            activation=activation,
            bias=bias
        )

        expected_tfc_block = TimeFrequencyConvolutionBlock(
            in_channels=in_channels,
            growth_rate=growth_rate,
            num_layers=num_layers,
            kernel_size=kernel_size,
            activation=activation,
            bias=bias
        )
        expected_tdf_block = TimeDistributedFullyConnectedBlock(
            channels=growth_rate,
            frequency_bins=frequency_bins,
            bottleneck=bottleneck,
            num_layers=num_layers,
            activation=activation,
            bias=bias
        )

        self.assertEqual(tfc_tdf_block.tfc_block.__str__(), 
                         expected_tfc_block.__str__())
        self.assertEqual(tfc_tdf_block.tdf_block.__str__(), 
                         expected_tdf_block.__str__())

    def test_forward_output_dims(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        bottleneck = 4
        activation = "ReLU"
        bias = False

        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")

        tfc_tdf_block = TFC_TDF_v1(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            bottleneck=bottleneck,
            activation=activation,
            bias=bias
        ).to(device)

        x = torch.rand([1, in_channels, 44, 1024]).to(device) # [batch, channel, n_frames, frequency_bins]
        output = tfc_tdf_block(x)
        expected_shape = [1, growth_rate, 44, 1024]

        self.assertEqual(list(output.shape), expected_shape)


class TestTFC_TDF_v1_Downsample(unittest.TestCase):
    def test_init_architecture(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        bottleneck = 4
        activation = "ReLU"
        bias = False

        tfc_tdf_downsample_block = TFC_TDF_v1_DownSample(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            bottleneck=bottleneck,
            activation=activation,
            bias=bias
        )

        expected_tfc_tdf_block = TFC_TDF_v1(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            bottleneck=bottleneck,
            activation=activation,
            bias=bias
        )

        expected_downsample_block = DownSample2DBlock(
            in_channels=growth_rate,
            out_channels=growth_rate,
            activation=activation,
            bias=bias
        )

        self.assertEqual(tfc_tdf_downsample_block.tfc_tdf_block.__str__(), 
                         expected_tfc_tdf_block.__str__())
        self.assertEqual(tfc_tdf_downsample_block.downsample.__str__(), 
                         expected_downsample_block.__str__())
    
    def test_forward_output_dims(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        bottleneck = 256
        activation = "ReLU"
        bias = False

        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")

        tfc_tdf_downsample_block = TFC_TDF_v1_DownSample(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            bottleneck=bottleneck,
            activation=activation,
            bias=bias
        ).to(device)

        input = torch.rand([1, in_channels, 44, 1024]).to(device)
        output = tfc_tdf_downsample_block(input)
        expected_shape = [1, growth_rate, 22, 512]

        self.assertEqual(list(output.shape), expected_shape)



class TestTFC_TDF_v1_UpSample(unittest.TestCase):
    def test_init_architecture(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        bottleneck = 256
        activation = "ReLU"
        bias = False

        tfc_tdf_downsample_block = TFC_TDF_v1_UpSample(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            bottleneck=bottleneck,
            activation=activation,
            bias=bias
        )

        expected_tfc_tdf_block = TFC_TDF_v1(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            bottleneck=bottleneck,
            activation=activation,
            bias=bias
        )

        expected_downsample_block = UpSample2DBlock(
            in_channels=growth_rate,
            out_channels=growth_rate,
            activation=activation,
            bias=bias
        )

        self.assertEqual(tfc_tdf_downsample_block.tfc_tdf_block.__str__(), 
                         expected_tfc_tdf_block.__str__())
        self.assertEqual(tfc_tdf_downsample_block.upsample.__str__(), 
                         expected_downsample_block.__str__())
    
    def test_forward_output_dims(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 512
        bottleneck = 256
        activation = "ReLU"
        bias = False

        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")

        tfc_tdf_downsample_block = TFC_TDF_v1_UpSample(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            bottleneck=bottleneck,
            activation=activation,
            bias=bias
        ).to(device)

        input = torch.rand([1, in_channels, 22, 512]).to(device)
        output = tfc_tdf_downsample_block(input)
        expected_shape = [1, growth_rate, 44, 1024]

        self.assertEqual(list(output.shape), expected_shape)


if __name__ == "__main__":
    unittest.main()